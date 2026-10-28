"""Feature capture copied from codex/n1-featcache; unchanged inference windows and clock."""
import numpy as np
from ensomi_n.features.mel import DEFAULT

def capture_song(model, song, device):
    """Observe the evaluator's linear call, stitching its tensors with the same edge rule.

    Returns float32 features, raw logits and the evaluator's original activations.
    The temporary hook is removed even when inference raises. No model state changes.
    """
    from ensomi_n.v0.infer import song_outputs, windows
    n = song.frames
    total = n + (-n) % 4
    starts = windows(total)
    feat = np.empty((total, model.heads.lin.in_features), dtype=np.float32)
    logits = np.empty((total, model.heads.lin.out_features), dtype=np.float32)
    best = np.full(total, -1.0)
    cursor = 0

    def observe(module, inputs, output):
        nonlocal cursor
        h, y = inputs[0].float().cpu().numpy(), output.float().cpu().numpy()
        length = y.shape[1]
        centre = np.minimum(np.arange(length), length - 1 - np.arange(length))
        for j in range(len(y)):
            start = starts[cursor]
            take = centre > best[start:start + length]
            idx = np.flatnonzero(take) + start
            feat[idx], logits[idx], best[idx] = h[j, take], y[j, take], centre[take]
            cursor += 1

    hook = model.heads.lin.register_forward_hook(observe)
    try:
        acts = song_outputs(model, song, device, batch=8)
    finally:
        hook.remove()
    if cursor != len(starts) or (best < 0).any() or len(acts['beat']) != n:
        raise ValueError(f'Feature clock/window mismatch: {song.item}')
    return feat[:n], logits[:n], acts

def evidence_from_logits(y):
    """Apply the existing head nonlinearities to stored raw logits on the CPU."""
    import torch
    from ensomi_n.v0.model import Heads, V0
    heads = Heads(y.shape[-1])
    heads.lin = torch.nn.Identity()
    with torch.inference_mode():
        return {k: v.numpy() for k, v in V0.activations(heads(torch.from_numpy(y.astype(np.float32)))).items()}

def differences(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('E2 evidence fields differ')
    result = {}
    for key, value in actual.items():
        if value.shape != expected[key].shape or not np.isfinite(value).all():
            raise ValueError(f'E2 evidence shape or finite-value mismatch: {key}')
        delta = np.abs(value.astype(np.float64) - expected[key])
        result[key] = float(delta.max(initial=0))
        if key in ('phase', 'bar_phase'):
            result[key + '_circular'] = float(np.minimum(delta, 1 - delta).max(initial=0))
    return result

def reproduce(sha, logits, acts, saved, e2):
    with np.load(e2 / f'{sha}.npz', allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    full = differences(acts, old)
    if not all(np.allclose(acts[k], old[k], atol=1e-6, rtol=1e-5) for k in old):
        raise ValueError(f'E2 float32 mismatch for {sha}: {full}')
    reconstructed = evidence_from_logits(logits)
    if not all(np.allclose(reconstructed[k], acts[k], atol=2e-6, rtol=1e-5) for k in acts):
        raise ValueError(f'Raw logit reconstruction mismatch for {sha}')
    with np.load(saved, allow_pickle=False) as z:
        stored = z['logits']
    if not np.array_equal(stored, logits.astype(np.float16)):
        raise ValueError(f'Stored logits differ from float16 conversion: {sha}')
    return dict(sha256=sha, full_precision_max_abs=full,
                cached_logits_evidence_max_abs=differences(evidence_from_logits(stored), old),
                raw_logit_float16_max_abs=float(np.abs(stored.astype(np.float32) - logits).max()))

def write_npz(path, feat, logits):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    arrays = dict(feat=feat.astype(np.float16), logits=logits.astype(np.float16),
                  t0_ms=np.float64(0), fps=np.float64(DEFAULT.fps))
    if not np.isfinite(arrays['feat']).all() or not np.isfinite(arrays['logits']).all():
        raise ValueError('Non-finite feature/logit after float16 conversion')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
