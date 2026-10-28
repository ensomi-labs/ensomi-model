"""Blind two-bar audio controls; identities are stored outside the packet."""
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from .common import ROOT, OUT, dump, labels, log, prereg_check, units


def native_audio(path):
    import soundfile as sf
    from ensomi_n.features.clock import file_lead
    try:
        x, sr = sf.read(path, dtype='float32', always_2d=True)
        x = x.mean(axis=1)
        decoder = 'libsndfile'
    except Exception:
        info = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate', '-of', 'csv=p=0', str(path)],
            check=True, capture_output=True, text=True)
        sr = int(info.stdout.strip().splitlines()[0].split(',')[0])
        out = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path),
            '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-'], check=True, capture_output=True)
        x = np.frombuffer(out.stdout, np.float32).copy()
        decoder = 'ffmpeg'
    lead, _, _ = file_lead(path, decoder)
    if lead:
        x = np.r_[np.zeros(lead, np.float32), x]
    return x, sr, decoder, lead


def render(x, sr, lo, hi, clicks):
    start, end = round(lo * sr / 1000), round(hi * sr / 1000)
    if start < 0 or end > len(x) or end <= start:
        raise ValueError('Snippet lies outside decoded audio')
    plain = x[start:end].copy()
    rms = float(np.sqrt(np.mean(plain.astype(float) ** 2)))
    if rms < 1e-6:
        raise ValueError('Silent control snippet')
    clicked = plain.copy()
    length = max(2, round(.002 * sr))
    wave = np.sin(2 * np.pi * 1500 * np.arange(length) / sr) * np.hanning(length)
    wave *= rms * 10 ** (-12 / 20) / max(np.sqrt(np.mean(wave ** 2)), 1e-12)
    for t in clicks:
        i = round((t - lo) * sr / 1000)
        if 0 <= i and i + length <= len(clicked):
            clicked[i:i + length] += wave
    together = np.r_[plain, np.zeros(round(.5 * sr), np.float32), clicked]
    peak = float(np.max(abs(together)))
    gain = min(1., .98 / max(peak, 1e-6))
    return (together * gain).astype('float32'), dict(source_rms=rms, common_gain=gain,
              click_samples=length, click_rms_ratio_db=-12., silence_samples=round(.5 * sr))


def controls():
    rng = np.random.default_rng(20261028)
    panel = [u for u in units() if u['cohort'] in ('panel', 'val_hard')]
    candidates = {0: [], 1: []}
    for u in panel:
        z = labels(u)
        for i, bar in enumerate(z['bar_ids'][:-1]):
            nextbar = z['bar_ids'][i + 1]
            if nextbar != bar + 1:
                continue
            q = np.flatnonzero(np.isin(z['bars'], [bar, nextbar]))
            if not len(q):
                continue
            lo, hi = z['beats'][q[0]], z['beats'][q[-1]] + z['length'][q[-1]]
            for cls in (0, 1):
                target = q[z['y'][q] == cls]
                if len(target):
                    candidates[cls].append(dict(unit=u, lo=float(lo), hi=float(hi),
                                                beats=target.tolist(), truth=cls, kind='control'))
    selected, recs = [], set()
    for cls in (1, 0):
        available = candidates[cls]
        for i in rng.permutation(len(available)):
            v = available[i]
            if v['unit']['rec'] in recs:
                continue
            if sum(x['truth'] == cls for x in selected) >= 8:
                break
            recs.add(v['unit']['rec']); selected.append(v)
    assert len(selected) == 16, 'Not enough distinct panel recordings for controls'
    return selected


def propagation_examples():
    """Reservoir-sample actual declared changes only after the both-seed gate."""
    from .report import propagation_verdict
    from .episodes import eligible, load_learned, simulate, song, update
    from .common import stable_seed
    verdict = propagation_verdict()
    arms = verdict['packet_eligible_arms']
    if not arms:
        return None
    arm = 'graph' if 'graph' in arms else sorted(arms)[0]
    cfg = json.loads((OUT / 'P7-config-s0.json').read_text())['configs']
    model = load_learned(0, int(arm[-1])) if arm.startswith('learned') else None
    reserved = {v['unit']['rec'] for v in controls()}
    rng = np.random.default_rng(20261028)
    reservoir, counts = {arm: [], 'random': []}, {arm: 0, 'random': 0}
    for u in units():
        if u['cohort'] not in ('panel', 'val_hard') or u['rec'] in reserved:
            continue
        s = song(0, u)
        regs = [r for r in s.regions if len(eligible(s.z, r))]
        for ri, r in enumerate(regs):
            for simulator in (0, 1):
                for position in range(4):
                    for run in (1, 3):
                        for confirming in (0, 8):
                            marks = simulate(s.z, s.p, r, s.c, simulator=simulator,
                                position=position, run_length=run, confirming=confirming, identity=(u['id'], ri))
                            if marks is None:
                                continue
                            for target_arm, cap in ((arm, 14), ('random', 10)):
                                p, loc, declared = update(s, marks, target_arm, cfg[target_arm],
                                    model if target_arm == arm else None,
                                    stable_seed((u['id'], ri, simulator, position, confirming)))
                                changed = declared[~loc[declared] & ~r[declared]]
                                for beat in changed:
                                    bar = s.z['bars'][beat]
                                    ix = int(np.searchsorted(s.z['bar_ids'], bar))
                                    if ix == len(s.z['bar_ids']) - 1:
                                        ix -= 1
                                    if ix < 0 or s.z['bar_ids'][ix + 1] != s.z['bar_ids'][ix] + 1:
                                        continue
                                    q = np.flatnonzero(np.isin(s.z['bars'], s.z['bar_ids'][ix:ix + 2]))
                                    value = dict(unit=u, lo=float(s.z['beats'][q[0]]),
                                        hi=float(s.z['beats'][q[-1]] + s.z['length'][q[-1]]),
                                        beats=[int(beat)], truth=int(s.z['y'][beat]), kind=target_arm,
                                        episode=dict(region=ri, simulator=simulator, position=position, run=run, confirming=confirming))
                                    counts[target_arm] += 1
                                    bag = reservoir[target_arm]
                                    if len(bag) < cap:
                                        bag.append(value)
                                    else:
                                        pick = rng.integers(counts[target_arm])
                                        if pick < cap:
                                            bag[pick] = value
    payload = dict(both_seed_pass=True, arm=arm, gate=verdict['predictions']['graph_over_random'] if arm == 'graph'
                   else verdict['learned'][arm]['over_random'], candidates=counts,
                   snippets=reservoir[arm] + reservoir['random'])
    dump(OUT / 'packet-propagation.json', payload)
    return payload


def build():
    import soundfile as sf
    prereg_check()
    selected = controls()
    recs = {v['unit']['rec'] for v in selected}
    rng = np.random.default_rng(20261028)
    payload = propagation_examples()
    propagation = []
    if payload is not None:
        assert payload['both_seed_pass'] is True
        propagation = [v for v in payload['snippets'] if v['unit']['rec'] not in recs][:24]
        selected.extend(propagation)
    rng.shuffle(selected)
    packet, keydir = OUT / 'packet', OUT / 'packet-key'
    packet.mkdir(parents=True, exist_ok=True); keydir.mkdir(parents=True, exist_ok=True)
    keys = []
    for i, v in enumerate(selected, 1):
        u = v['unit']; z = labels(u)
        x, sr, decoder, lead = native_audio(ROOT / u['audio'])
        clicks = [float(z['beats'][b] + z['length'][b] * k / 3) for b in v['beats'] for k in (1, 2)]
        wave, stats = render(x, sr, v['lo'], v['hi'], clicks)
        name = f'{i:02d}'
        sf.write(packet / (name + '.wav'), wave, sr, subtype='PCM_16')
        keys.append(dict(id=name, audio_sha256=u['sha'], rec=u['rec'], audio=u['audio'],
                         unit=u['id'], lo_ms=v['lo'], hi_ms=v['hi'],
                         target_beats_ms=z['beats'][v['beats']].tolist(),
                         chart_class={1: 'ternary', 0: 'binary-fine', -1: 'unlabelled'}[v['truth']],
                         kind=v['kind'], decoder=decoder, lead_samples=lead,
                         episode=v.get('episode'),
                         native_rate=sr, seconds=len(wave) / sr, **stats))
    with (packet / 'answers.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'answer', 'note'])
        w.writerows([[k['id'], '', ''] for k in keys])
    message = ('This packet contains 16 chart-label controls only. No propagation arm qualified '
               'under the both-seed rule. It is shorter than the full 25–30 minute packet.'
               if not propagation else 'This packet includes chart-label controls and propagation examples.')
    text = f'''# Listening packet

{message}

Listen to the numbered WAV files in order. Each plays two bars plain, then pauses
for half a second and plays the same audio with soft clicks. Replay as needed.

**In the second play, do the clicks sit on notes you hear as a three-feel?**

Enter `yes`, `no`, or `unsure` in `answers.csv`. Add a note if useful. Judge the
music you hear; the chart labels have not been validated by a listener. Keep the
identity key closed until answers are saved.

There are {len(keys)} clips, totaling {sum(k['seconds'] for k in keys) / 60:.1f} minutes
before replays. Each clip's two plays have exactly the same source audio and gain.
'''
    (packet / 'packet.md').write_text(text)
    dump(keydir / 'key.json', dict(seed=20261028, composition=dict(
        controls_ternary=8, controls_binary_fine=8, propagation=len(propagation),
        propagation_arms=dict(__import__('collections').Counter(v['kind'] for v in propagation))), snippets=keys))
    log('packet_done', path=str(packet), clips=len(keys), seconds=sum(k['seconds'] for k in keys))
    return keys
