from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import LABELS, MASK, POWERS, ROWS, TAGS, collate, digest, prepare, replay
from .model import Denoiser, masked_loss, sample


def emit(out, event, **values):
    row = {"event": event, **values}
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    print(text, flush=True)
    with (out / "metrics.jsonl").open("a") as handle:
        handle.write(text + "\n")
    temp = out / "status.tmp"
    temp.write_text(text + "\n")
    temp.replace(out / "status.json")


@torch.no_grad()
def evaluate(model, samples, device):
    model.eval()
    records = []
    for item in samples:
        batch = collate([item], device)
        unknown = batch["target"].new_full(batch["target"].shape, MASK)
        s = torch.ones(1, device=device)
        context = model.context(batch)
        logits = model(unknown, s, batch, context)
        target = batch["target"] - 1
        loss = F.cross_entropy(logits.transpose(1, 2), target).item()
        prediction = logits.argmax(-1)[0].cpu().numpy() + 1
        batch["style"] = torch.zeros_like(batch["style"])
        unconditioned = F.cross_entropy(model(unknown, s, batch, context).transpose(1, 2), target).item()
        try:
            legal = replay(prediction, item["entry"]) == item["exit"]
        except ValueError:
            legal = False
        records.append({"id": item["id"], "tag": TAGS[item["tag"]], "label": LABELS[item["label"]],
                        "rows": item["rows"], "full_mask_ce": loss,
                        "condition_nll_margin": unconditioned - loss,
                        "row_accuracy": float((prediction == item["rows_code"]).mean()),
                        "lane_accuracy": float((ROWS[prediction - 1] == ROWS[item["rows_code"] - 1]).mean()),
                        "raw_argmax_boundary_legal": bool(legal)})
    metrics = {key: float(np.mean([r[key] for r in records])) for key in
               ("full_mask_ce", "condition_nll_margin", "row_accuracy", "lane_accuracy", "raw_argmax_boundary_legal")}
    return {"scopes": len(records), **metrics}, records


def export_osu(item, generated, destination):
    """Replace only the gold scope, retaining source entry/exit LN occupancy."""
    times, codes = item["all_times"], item["all_codes"].copy()
    lo, hi = item["scope"]["startMs"], item["scope"]["endMs"]
    inside = (times >= lo) & (times < hi)
    codes[inside] = generated
    if replay(codes) != 0:
        raise ValueError("Exported whole chart is not legal")
    objects, holds = [], {}
    for timestamp, code in zip(times, codes):
        for lane, action in enumerate(ROWS[code - 1]):
            x = 64 + lane * 128
            if action == 1:
                objects.append((int(timestamp), f"{x},192,{timestamp},1,0,0:0:0:0:"))
            elif action == 2:
                holds[lane] = int(timestamp)
            elif action == 3:
                start = holds.pop(lane)
                objects.append((start, f"{x},192,{start},128,0,{timestamp}:0:0:0:0:"))
    original = Path(item["source_path"]).read_text(encoding="utf-8-sig")
    header = original.split("[HitObjects]", 1)[0]
    audio = Path(item["audio_path"])
    link = destination.parent / ("audio" + audio.suffix)
    if not link.exists():
        link.symlink_to(audio)
    lines = []
    for line in header.splitlines():
        if line.startswith("AudioFilename:"):
            line = "AudioFilename: " + link.name
        elif line.startswith("Version:"):
            line = "Version: Gold diffusion " + destination.stem
        elif line.startswith(("BeatmapID:", "BeatmapSetID:")):
            line = line.split(":")[0] + ":-1"
        lines.append(line)
    destination.write_text("\n".join(lines) + "\n[HitObjects]\n" + "\n".join(s for _, s in sorted(objects)) + "\n")


def generate_review(model, samples, train, config, out, split):
    cells = sorted({(s["tag"], s["label"]) for s in train})
    records = []
    for item in samples[:config.sample_scopes]:
        # Cover each positively supervised tag; never request an untrained cell.
        own = (item["tag"], item["label"])
        alternatives = [(tag, max(label for t, label in cells if t == tag and label >= 2))
                        for tag in range(len(TAGS)) if any(t == tag and label >= 2 for t, label in cells)]
        controls = list(dict.fromkeys(([own] if own in cells else []) + alternatives))
        for tag, label in controls:
            for seed in range(config.sample_seeds):
                batch = collate([item], config.device)
                batch["style"].zero_()
                batch["style"][0, tag] = label
                generated = sample(model, batch, item["entry"], item["exit"], config.sampling_steps,
                                   config.seed + seed, config.temperature)
                directory = out / "samples" / split / item["id"].replace(":", "_")
                directory.mkdir(parents=True, exist_ok=True)
                filename = f"{TAGS[tag]}-{LABELS[label]}-seed{seed}.osu"
                export_osu(item, generated, directory / filename)
                records.append({"scope_id": item["id"], "source": item["source"], "scope": item["scope"],
                                "requested_style": {TAGS[tag]: LABELS[label]}, "seed": config.seed + seed,
                                "path": str((directory / filename).relative_to(out)),
                                "times_ms": item["times_ms"].tolist(), "row_codes": generated.tolist(),
                                "reference_row_agreement": float((generated == item["rows_code"]).mean()),
                                "boundary_legal": True, "human_style_judgment": None})
    (out / f"{split}_samples.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    return records


def snapshot(out):
    directory = Path(__file__).parent
    target = out / "source_snapshot"
    target.mkdir()
    files = list(directory.glob("*.py"))
    files.append(directory.parents[1] / "configs/hydra/gold_diffusion.yaml")
    for path in files:
        shutil.copy2(path, target / path.name)
    info = {"files": {p.name: digest(p) for p in files},
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "working_tree": subprocess.check_output(["git", "status", "--short"], text=True),
            "torch": torch.__version__, "device": "mps" if torch.backends.mps.is_available() else "cpu"}
    (out / "provenance.json").write_text(json.dumps(info, indent=2))


def train(config, out):
    torch.set_num_threads(config.cpu_threads)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; select device=cpu explicitly")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    snapshot(out)
    emit(out, "preparing_gold", device=config.device)
    samples, manifest = prepare(config, out)
    torch.save(samples, out / "data_snapshot.pt")
    splits = {s: [i for i in samples if i["split"] == s] for s in ("train", "validation", "test")}
    model = Denoiser(config.width, config.heads, config.layers, config.dropout).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    emit(out, "training_started", counts=manifest["counts"], excluded=len(manifest["excluded"]),
         parameters=sum(p.numel() for p in model.parameters()))
    initial, records = evaluate(model, splits["validation"], config.device)
    (out / "validation_initial.json").write_text(json.dumps(records, indent=2))
    emit(out, "validation", step=0, **initial)
    best, stale, step = initial["full_mask_ce"], 0, 0
    torch.save({"model": model.state_dict(), "config": asdict(config), "step": 0}, out / "best.pt")
    start = time.monotonic()
    accumulated = []
    reason = "steps"
    for step in range(1, config.steps + 1):
        model.train()
        indices = rng.choice(len(splits["train"]), size=config.batch_size, replace=True)
        batch = collate([splits["train"][i] for i in indices], config.device)
        drop = torch.rand(config.batch_size, generator=generator) < config.condition_dropout
        batch["style"][drop.to(config.device)] = 0
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(model, batch, generator)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        accumulated.append(loss.item())
        if step % config.log_every == 0:
            emit(out, "train", step=step, loss=float(np.mean(accumulated)),
                 elapsed_seconds=round(time.monotonic() - start, 2),
                 mps_allocated_mb=round(torch.mps.current_allocated_memory() / 2**20, 1) if config.device == "mps" else None)
            accumulated.clear()
        timeout = time.monotonic() - start >= config.max_seconds
        if step % config.eval_every == 0 or step == config.steps or timeout:
            metrics, records = evaluate(model, splits["validation"], config.device)
            (out / "validation_latest.json").write_text(json.dumps(records, indent=2))
            emit(out, "validation", step=step, **metrics)
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "config": asdict(config), "step": step}
            torch.save(payload, out / "last.pt")
            if metrics["full_mask_ce"] < best:
                best, stale = metrics["full_mask_ce"], 0
                torch.save(payload, out / "best.pt")
            else:
                stale += 1
            if timeout or stale >= config.patience:
                reason = "wall_time" if timeout else "validation_patience"
                break
    checkpoint = torch.load(out / "best.pt", map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics, _ = evaluate(model, splits["validation"], config.device)
    generated = generate_review(model, splits["validation"], splits["train"], config, out, "validation")
    result = {"best_step": checkpoint["step"], "last_step": step, "stop_reason": reason,
              "validation": metrics, "validation_samples": len(generated),
              "semantic_style_adherence": "requires independent human review"}
    if config.final_test:
        metrics, records = evaluate(model, splits["test"], config.device)
        (out / "test_metrics.json").write_text(json.dumps(records, indent=2))
        result["test"] = metrics
        result["test_samples"] = len(generate_review(model, splits["test"], splits["train"], config, out, "test"))
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    emit(out, "completed", **result)
