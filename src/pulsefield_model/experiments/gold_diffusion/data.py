from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from pulsefield_model.features.mel import load_cached_music_log_mel, music_log_mel_cache_path
from pulsefield_model.features.mel_base import MUSIC_MEL_CACHE_CONFIG

TAGS = ("jack-organization", "stream-organization", "trill-organization", "tech", "ln-coordination")
LABELS = ("unknown", "absent", "supporting", "prominent")
MASK = 256
POWERS = np.array([1, 4, 16, 64], dtype=np.int64)
ROWS = (np.arange(1, 256)[:, None] // POWERS) % 4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settled_observations(document):
    """Match Lens latest-decision semantics; machine audits never create gold."""
    latest = {(d["handoffId"], d["claimId"]): d for d in document["decisions"]}
    for observation in document["observations"]:
        origin = observation["origin"]
        if origin["kind"] == "agent-proposal":
            decision = latest.get((origin["handoffId"], origin["claimId"]))
            if not decision or decision["disposition"] not in ("accepted", "modified"):
                continue
            if decision.get("observationId") != observation["id"]:
                continue
            if decision["humanId"] != observation["humanId"]:
                raise ValueError("Observation human identity differs from decision")
        elif origin["kind"] != "direct-human":
            raise ValueError(f"Unknown observation origin: {origin}")
        claim = observation["claim"]
        if claim["assessment"]["presence"] not in ("present", "absent"):
            continue
        if claim["tagId"] not in TAGS:
            raise ValueError(f"Unknown gold taxonomy: {claim['tagId']}")
        if not observation["humanId"] or not observation["foundationSha256"]:
            raise ValueError("Gold requires human and Foundation provenance")
        yield observation


def replay(codes, initial=0):
    state = initial
    for code in codes:
        if not 1 <= int(code) <= 255:
            raise ValueError("Replay requires nonempty materialized rows")
        for lane, action in enumerate(ROWS[int(code) - 1]):
            opened = bool(state & (1 << lane))
            if (action in (1, 2) and opened) or (action == 3 and not opened):
                raise ValueError(f"Illegal action {action} on lane {lane}")
            if action == 2:
                state |= 1 << lane
            elif action == 3:
                state &= ~(1 << lane)
    return state


def action_rows(notes):
    by_time = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for note in notes:
        lane = int(note["column"])
        events = [(int(note["start_ms"]), 2 if note["kind"] == "long" else 1)]
        if note["kind"] == "long":
            events.append((int(note["end_ms"]), 3))
        for time, action in events:
            if by_time[time][lane]:
                raise ValueError("Multiple actions on a lane at one timestamp")
            by_time[time][lane] = action
    times = np.array(sorted(by_time), dtype=np.int64)
    codes = np.array([by_time[t] @ POWERS for t in times], dtype=np.int64)
    if replay(codes) != 0:
        raise ValueError("Source leaves open long notes")
    return times, codes


def audio_path(source):
    chart = Path(source["sourcePath"])
    for line in chart.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("AudioFilename:"):
            path = chart.parent / line.split(":", 1)[1].strip()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path.resolve()
    raise ValueError(f"No AudioFilename in {chart}")


def split_groups(examples, seed, fraction):
    """Song-group split, retaining at least one training group per observed cell."""
    groups = defaultdict(list)
    for item in examples:
        groups[item["group"]].append(item)
    cells = lambda items: {(i["tag"], i["label"]) for i in items}
    counts = Counter(c for items in groups.values() for c in cells(items))
    ordered = sorted(groups)
    np.random.default_rng(seed).shuffle(ordered)
    assignment = {g: "train" for g in groups}
    count = max(1, round(len(groups) * fraction))
    for split in ("validation", "test"):
        taken = 0
        for group in ordered:
            if assignment[group] != "train":
                continue
            present = cells(groups[group])
            if all(counts[c] > 1 for c in present):
                assignment[group] = split
                counts.subtract(present)
                taken += 1
            if taken == count:
                break
        if not taken:
            raise ValueError("Too few independent gold groups for train/validation/test")
    for item in examples:
        item["split"] = assignment[item["group"]]


def prepare(config, out):
    lens = Path(config.lens_root).resolve()
    campaign = lens / ".local/corpus-500"
    workflow = lens / ".local/corpus-500-v2/workspace/workflow"
    sources = {s["source"]["sha256"]: s for s in json.loads((campaign / "admin/source-map.json").read_text())}
    raw, hashes = [], {}
    for path in sorted(workflow.glob("*.v2.json")):
        content = path.read_bytes()
        doc = json.loads(content)["document"]
        observations = list(settled_observations(doc))
        if not observations:
            continue
        sha = path.name.removesuffix(".v2.json")
        hashes[path.name] = hashlib.sha256(content).hexdigest()
        for observation in observations:
            claim = observation["claim"]
            label = claim["assessment"].get("salience") if claim["assessment"]["presence"] == "present" else "absent"
            if label not in LABELS[1:]:
                raise ValueError("Settled positive gold requires ordinal salience")
            raw.append({"id": observation["id"], "source": sha,
                        "tag": TAGS.index(claim["tagId"]), "label": LABELS.index(label),
                        "scope": claim["scope"], "human": observation["humanId"],
                        "confirmed_at": observation["confirmedAt"],
                        "foundation": observation["foundationSha256"]})
    if not raw:
        raise ValueError(f"No settled human observations in {workflow}")
    examples, rejected, chart_cache, fingerprints = [], [], {}, {}
    seen = {}
    # Conflicting exact cells are not silently chosen by timestamp or duplicated.
    for item in raw:
        key = (item["source"], item["scope"]["startMs"], item["scope"]["endMs"], item["tag"])
        if key in seen:
            if seen[key]["label"] != item["label"]:
                raise ValueError(f"Conflicting settled gold for {key}")
            rejected.append({"id": item["id"], "reason": "duplicate exact gold cell"})
            continue
        seen[key] = item
        sha = item["source"]
        if sha not in chart_cache:
            source = sources[sha]
            if digest(source["sourcePath"]) != sha:
                raise ValueError(f"Source bytes changed: {sha}")
            parquet = campaign / "agent/charts" / f"{sha}.parquet"
            table = pq.read_table(parquet)
            if json.loads(table.schema.metadata[b"beatmap_lens"])["source"]["sha256"] != sha:
                raise ValueError("Parquet source mismatch")
            times, codes = action_rows(table.to_pylist())
            audio = audio_path(source)
            audio_sha = digest(audio)
            chart_cache[sha] = (times, codes, source, audio, audio_sha)
            fingerprints[sha] = {"parquet_sha256": digest(parquet), "audio_sha256": audio_sha,
                                 "source_path": source["sourcePath"], "audio_path": str(audio)}
        times, codes, source, audio, audio_sha = chart_cache[sha]
        lo, hi = item["scope"]["startMs"], item["scope"]["endMs"]
        selected = (times >= lo) & (times < hi)
        size = int(selected.sum())
        if not 1 <= size <= config.max_rows:
            rejected.append({"id": item["id"], "reason": "row count outside configured limit", "rows": size})
            continue
        before = times < lo
        entry = replay(codes[before])
        exit_state = replay(codes[selected], entry)
        # Exact audio hash groups encodings of the same file; artist/title additionally
        # joins mapsets with separately encoded copies of the same named song.
        song = (source["source"]["artist"].casefold().strip(), source["source"]["title"].casefold().strip())
        item.update(rows=size, group=json.dumps(song), audio_sha256=audio_sha,
                    entry=entry, exit=exit_state, title=source["source"]["title"])
        examples.append(item)
    # Union groups sharing audio bytes even if their metadata differs.
    aliases = {}
    for item in examples:
        a, b = item["group"], aliases.get(item["audio_sha256"], item["group"])
        if a != b:
            for other in examples:
                if other["group"] == a:
                    other["group"] = b
            aliases = {k: b if v == a else v for k, v in aliases.items()}
        aliases[item["audio_sha256"]] = b
    split_groups(examples, config.seed, config.holdout_fraction)
    manifest = {"gold": examples, "excluded": rejected, "workflow_sha256": hashes,
                "sources": fingerprints, "tags": TAGS, "labels": LABELS,
                "mel_config_hash": MUSIC_MEL_CACHE_CONFIG.mel_config_hash,
                "source_map_sha256": digest(campaign / "admin/source-map.json"),
                "counts": dict(Counter(i["split"] for i in examples)),
                "cells": {s: dict(Counter(f"{TAGS[i['tag']]}:{LABELS[i['label']]}" for i in examples if i["split"] == s))
                          for s in ("train", "validation", "test")}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    samples, mel_cache, mel_hashes = [], {}, {}
    for item in examples:
        times, codes, source, audio, _ = chart_cache[item["source"]]
        if audio not in mel_cache:
            try:
                mel_cache[audio] = load_cached_music_log_mel(audio, mmap_mode="r")
            except FileNotFoundError:
                from pulsefield_model.features.audio import load_audio_file
                from pulsefield_model.features.mel_base import compute_log_mel_10ms
                waveform = load_audio_file(audio, sample_rate=MUSIC_MEL_CACHE_CONFIG.sample_rate, normalize=True)
                mel = compute_log_mel_10ms(waveform, sample_rate=24000, config=MUSIC_MEL_CACHE_CONFIG)
                cache_path = music_log_mel_cache_path(audio)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".gold-diffusion.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, mel)
                temporary.replace(cache_path)
                mel_cache[audio] = mel
            mel_hashes[audio] = digest(music_log_mel_cache_path(audio))
        fingerprints[item["source"]].update(mel_sha256=mel_hashes[audio],
                                             mel_path=str(music_log_mel_cache_path(audio).resolve()))
        lo, hi = item["scope"]["startMs"], item["scope"]["endMs"]
        target = (times >= lo) & (times < hi)
        history = times < lo
        style = np.zeros(5, dtype=np.int64)
        style[item["tag"]] = item["label"]
        sample = dict(item, times_ms=times[target], rows_code=codes[target],
                      history_codes=codes[history], history_times=(times[history] - lo) / 1000,
                      times=(times[target] - lo) / 1000,
                      mel=mel_patches(mel_cache[audio], times[target]), style=style,
                      all_times=times, all_codes=codes, source_path=source["sourcePath"], audio_path=str(audio))
        samples.append(sample)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return samples, manifest


def mel_patches(mel, times_ms):
    query = (np.asarray(times_ms)[:, None] + np.arange(-200, 201, 10)[None, :] - 20) / 10
    lo = np.floor(query).astype(np.int64)
    fraction = query - lo
    def read(indices):
        valid = (indices >= 0) & (indices < len(mel))
        values = np.asarray(mel[np.clip(indices, 0, len(mel) - 1)])
        return np.where(valid[..., None], values, np.log(1e-5))
    return ((1 - fraction[..., None]) * read(lo) + fraction[..., None] * read(lo + 1)).astype(np.float32)


def collate(samples, device):
    b, k, n = len(samples), max(s["rows"] for s in samples), max(1, max(len(s["history_codes"]) for s in samples))
    result = {"target": np.zeros((b, k), np.int64), "times": np.zeros((b, k), np.float32),
              "valid": np.zeros((b, k), bool), "history": np.zeros((b, n), np.int64),
              "history_times": np.zeros((b, n), np.float32), "history_valid": np.zeros((b, n), bool),
              "mel": np.zeros((b, k, 41, 128), np.float32), "style": np.stack([s["style"] for s in samples]),
              "entry": np.array([[(s["entry"] >> lane) & 1 for lane in range(4)] for s in samples], np.float32)}
    for i, sample in enumerate(samples):
        size, length = sample["rows"], len(sample["history_codes"])
        for key, source in (("target", "rows_code"), ("times", "times"), ("mel", "mel")):
            result[key][i, :size] = sample[source]
        result["valid"][i, :size] = True
        result["history"][i, :length] = sample["history_codes"]
        result["history_times"][i, :length] = sample["history_times"]
        result["history_valid"][i, :length] = True
    return {key: torch.as_tensor(value, device=device) for key, value in result.items()}
