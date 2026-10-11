"""Copy path on frozen R2, re-keyed for generated rows: a 15-mask bias through R2's ``head_mask_bias`` contract.

    bias_k(m) = (b_c - 1) lc_k(m) + (b_l - 1) ll_k(m) + sum_{h,t} w_{h,t}(g_k) C_{h,t,k}(m)
    C_{h,t,k} = sum_{j allowed} a_{h,k,j} V_t(h_j)

lc, ll: R2's log count marginal and log lanes-given-count at row k; V_t one-hot of the earlier row's mask, of its
mirror, lane overlap with it, and same head count; a_{h,k,.} a softmax over the allowed earlier rows j and a null
slot of an MLP of the relation between rows k and j; the gate w and the null logit read R2's mask distribution.

The relation (``CopyPath.scores``), from the 2026-10-09 path (``.sync/cp/scratch/r2-copy/copypath.py``) with three
changes for rows that come from audio:

- **Rhythm context with tolerance.** The 9-entry key (the 8 inter-onset intervals around the row in 1/48 beat and
  the beat phase) matches entry by entry within ``key_tol`` ticks (phase circularly); the pattern of matching
  entries (512 codes) is embedded, so partial matches carry their own weight. ``key_tol`` 0 is the old exact key.
- **Sound.** Two similarities of the rows' audio (``row_audio``): the sound at the row (log-Mel, 32 bands,
  [t - 20, t + 80) ms) and its 2-beat context ([b - 1, b + 1) beats), each centred on the song and compared
  across bands (Pearson). They enter as a continuous pair and as 8 buckets each.
- **No copy from the last rows.** Rows at most ``exclude_rows`` rows or ``exclude_beats`` beats back are never
  attended (the old path's likelihood gain sat at 4-32 beats; row lags 1-2 and beat lags 1-2 carried none).

The jack dial (``jack_dial``) acts at sampling time only. ``strength`` s in [0, 1] sets the row's jack mass (the
mass of masks sharing a lane with the previous row) to J_R2 + s (J_copy - J_R2): s = 1 is the trained path, s = 0
leaves R2's jack mass and lets the copy act only within jack and within non-jack masks. Monotone in s per row.

Everything is computed from committed masks, supplied head times and the song's audio, so the same function
serves teacher forcing and free running (``CopyBias``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn

COUNT = np.array([int(m).bit_count() for m in range(16)])
MIRROR = np.array([sum(((m >> j) & 1) << (3 - j) for j in range(4)) for m in range(16)])
OVERLAP = COUNT[np.bitwise_and(np.arange(16)[:, None], np.arange(1, 16)[None])] / 4.0      # [16,15]
SAMECOUNT = (COUNT[:, None] == COUNT[None, 1:]).astype(np.float64)                         # [16,15]
LAGS = (0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
NVALUES = 4
BANDS = 32                      # audio bands: 128 log-Mel bins averaged in fours
SOUND_MS = (-20.0, 80.0)        # the window of a row's own sound
AUDIO_BUCKETS = 8


@dataclass
class CopyConfig:
    heads: int = 8
    hidden: int = 32
    key_tol: int = 1                # rhythm-context entries match within this many 1/48 beat (0: exact)
    audio: bool = True              # sound and context similarity of the two rows
    exclude_rows: int = 2           # attend only to rows more than this many rows back ...
    exclude_beats: float = 2.0      # ... and more than this many beats back (0: no beat condition)

    @classmethod
    def old(cls):
        """The 2026-10-09 'full' path: exact key, no audio, every earlier row."""
        return cls(key_tol=0, audio=False, exclude_rows=0, exclude_beats=0.0)


def timing_keys(b, n):
    """[n, 9] int: inter-onset intervals k-4..k+3 in 1/48 beats (-1 where undefined) and beat phase."""
    ioi = np.rint(48 * np.diff(b)).astype(np.int64)
    keys = np.full((n, 9), -1, dtype=np.int64)
    for col, off in enumerate(range(-4, 4)):
        i = np.arange(n) + off
        ok = (i >= 0) & (i < len(ioi))
        keys[ok, col] = ioi[i[ok]]
    keys[:, 8] = np.rint(48 * np.mod(b[:n], 1)).astype(np.int64) % 48
    return keys


def key_code(keyq, keyk, tol: int):
    """[..., 9] keys of query and key rows (broadcast) -> pattern code 0..511 of the entries that match.

    An interval entry matches when both are defined and differ by at most ``tol``; the phase entry matches
    when its circular distance on 48 is at most ``tol``."""
    diff = (keyq - keyk).abs()
    ioi = (diff[..., :8] <= tol) & (keyq[..., :8] >= 0) & (keyk[..., :8] >= 0)
    phase = torch.minimum(diff[..., 8:], 48 - diff[..., 8:]) <= tol
    eq = torch.cat((ioi, phase), -1)
    return (eq.long() << torch.arange(9, device=eq.device)).sum(-1)


def _pearson_rows(x):
    x = x - x.mean(-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def row_audio(mel, head_ms, grid) -> np.ndarray:
    """[K, 2, BANDS] float32 unit vectors: each row's sound and its 2-beat context.

    ``mel`` [F, >=128] standardised log-Mel at 10 ms (frame i covers [10 i, 10 i + 40) ms), on the clock of
    ``head_ms``. Bands are centred on the song, then each vector is centred and normalised across bands."""
    m = np.asarray(mel[:, :128], dtype=np.float32).reshape(len(mel), BANDS, 4).mean(2)
    m = m - m.mean(0)
    cs = np.concatenate((np.zeros((1, BANDS), np.float32), np.cumsum(m, 0)))
    F = len(m)
    head_ms = np.asarray(head_ms, dtype=np.float64)

    def window(a_ms, b_ms):
        a = np.clip(np.floor(a_ms / 10.0).astype(np.int64), 0, F - 1)
        b = np.clip(np.ceil(b_ms / 10.0).astype(np.int64), a + 1, F)
        return (cs[b] - cs[a]) / (b - a)[:, None]

    beats = grid.beat(head_ms)
    sound = window(head_ms + SOUND_MS[0], head_ms + SOUND_MS[1])
    context = window(grid.time_of_beat(beats - 1.0), grid.time_of_beat(beats + 1.0))
    return np.stack((_pearson_rows(sound), _pearson_rows(context)), 1).astype(np.float32)


def gate_features(lp):
    """[..., 7] from R2's log mask marginal: 1, count marginal (4), normalised entropy, max probability."""
    p = torch.softmax(lp, -1)
    onehot = nn.functional.one_hot(torch.as_tensor(COUNT[1:] - 1, device=lp.device), 4).to(p.dtype)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1, keepdim=True) / math.log(15)
    return torch.cat([torch.ones_like(ent), p @ onehot, ent, p.max(-1, keepdim=True).values], -1)


def source_parts(lp):
    """lp [..., 15] -> lc, ll [..., 15]: log count marginal and log lanes given count (0 where lp is -inf)."""
    p = torch.softmax(lp, -1)
    onehot = nn.functional.one_hot(torch.as_tensor(COUNT[1:] - 1, device=lp.device), 4).to(p.dtype)
    lc = torch.log((p @ onehot).clamp_min(1e-30)) @ onehot.T
    ll = torch.log_softmax(lp, -1) - lc
    return lc, torch.where(torch.isfinite(ll), ll, torch.zeros_like(ll))


def _bucket(s):
    return ((s.clamp(-1, 1 - 1e-6) + 1) * (AUDIO_BUCKETS / 2)).long()


class CopyPath(nn.Module):
    def __init__(self, cfg: CopyConfig = CopyConfig()):
        super().__init__()
        self.cfg = cfg
        hidden, heads = cfg.hidden, cfg.heads
        self.register_buffer('lags', torch.tensor(LAGS))
        self.register_buffer('values', torch.tensor(np.stack([np.eye(16)[:, 1:], np.eye(16)[MIRROR][:, 1:], OVERLAP,
                                                              SAMECOUNT], 1), dtype=torch.float32))   # [16,4,15]
        self.e_lag = nn.Embedding(len(LAGS) + 1, hidden)
        self.e_align = nn.Embedding(4, hidden)
        self.e_row = nn.Embedding(512, hidden)
        self.e_key = nn.Embedding(512, hidden)
        self.w_cont = nn.Linear(2, hidden)
        if cfg.audio:
            self.w_audio = nn.Linear(2, hidden)
            self.e_sound = nn.Embedding(AUDIO_BUCKETS, hidden)
            self.e_context = nn.Embedding(AUDIO_BUCKETS, hidden)
        self.out = nn.Linear(hidden, heads)
        self.null = nn.Linear(7, heads)
        self.gate = nn.Linear(7, heads * NVALUES)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.beta = nn.Parameter(torch.ones(2))

    def relations(self, bq, bk, rq, jk, keyq, keyk, aq=None, ak=None):
        """Beat distance d, row distance r [B,q,J], key codes and audio similarities [B,q,J,2] (or None)."""
        d = bq[:, :, None] - bk[:, None, :]
        r = rq[:, :, None] - jk[:, None, :]
        code = key_code(keyq[:, :, None, :], keyk[:, None, :, :], self.cfg.key_tol)
        sim = torch.einsum('bqcf,bjcf->bqjc', aq, ak) if self.cfg.audio else None
        return d, r, code, sim

    def scores(self, d, r, code, sim):
        """Relation scores [B,q,J,heads]."""
        rc = r.clamp_min(1)
        lr = torch.log2(rc.float())
        h = self.e_row(rc.clamp_max(512) - 1) + self.w_cont(torch.stack([torch.log2(d.clamp_min(1 / 48)) / 7, lr / 11], -1))
        hit = (d[..., None] - self.lags).abs() <= .008
        lag = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(rc, len(LAGS)))
        q4 = d / 4
        align = ((q4 - q4.round()).abs() <= .002).long() * 2 + ((d - d.round()).abs() <= .008).long()
        h = h + self.e_lag(lag) + self.e_align(align) + self.e_key(code)
        if self.cfg.audio:
            h = h + self.w_audio(sim) + self.e_sound(_bucket(sim[..., 0])) + self.e_context(_bucket(sim[..., 1]))
        return self.out(nn.functional.gelu(h))

    def allowed(self, d, r, valid):
        ok = (r > max(self.cfg.exclude_rows, 0)) & valid[:, None, :]
        if self.cfg.exclude_beats > 0:
            ok &= d > self.cfg.exclude_beats + .008
        return ok

    def bias(self, rows, bq, kq, aq, lp, h, bk, kk, ak, valid, use_copy=True, drop=(), trace=None):
        """Query side: rows [B,q] long, bq [B,q] beats, kq [B,q,9] keys, aq [B,q,2,BANDS] audio (None without
        audio), lp [B,q,15] R2 log mask marginal. Key side (rows 0..J-1, padded): h [B,J] masks, bk [B,J] beats,
        kk [B,J,9] keys, ak [B,J,2,BANDS], valid [B,J] bool. Returns the mask bias [B,q,15]; with ``trace`` (a
        dict) also the attention [B,q,J+1,heads] and the relations."""
        lc, ll = source_parts(lp)
        out = (self.beta[0] - 1) * lc + (self.beta[1] - 1) * ll
        J = h.shape[1]
        if J == 0 or not use_copy:
            return out
        jk = torch.arange(J, device=rows.device)[None].expand(len(rows), -1)
        d, r, code, sim = self.relations(bq, bk, rows, jk, kq, kk, aq, ak)
        s = self.scores(d, r, code, sim)
        allowed = self.allowed(d, r, valid)
        s = s.masked_fill(~allowed[..., None], -torch.inf)
        g = gate_features(lp)
        a = torch.softmax(torch.cat([s, self.null(g)[:, :, None, :]], 2), 2)
        if trace is not None:
            trace.update(attention=a, d=d, r=r, code=code, sim=sim, allowed=allowed)
        V = self.values[h]                                                                 # [B,J,4,15]
        C = torch.einsum('bqjh,bjtm->bqhtm', a[:, :, :J], V)
        w = self.gate(g).view(*g.shape[:2], self.cfg.heads, NVALUES)
        if drop:
            keep = torch.ones(NVALUES, device=w.device)
            keep[list(drop)] = 0
            w = w * keep
        return out + torch.einsum('bqhtm,bqht->bqm', C, w)

    def chart_bias(self, rows, h, b, keys, aud, lp, pad=1, **kw):
        """One chart: rows [q] sorted, h masks of rows < max(rows) (at least), b [n'] beats, keys [n', 9] and aud
        [n', 2, BANDS] (or None) with n' > max(rows), lp [q, 15]. Keys are padded to a multiple of pad. -> [q, 15]."""
        J = int(rows.max()) if len(rows) else 0
        Jp = max(-(-J // pad) * pad, 1)
        j = torch.arange(Jp, device=rows.device)
        idx = j.clamp_max(max(J - 1, 0))
        a = aud if self.cfg.audio else None
        return self.bias(rows[None], b[rows][None], keys[rows][None], None if a is None else a[rows][None], lp[None],
                         h[idx][None], b[idx][None], keys[idx][None], None if a is None else a[idx][None],
                         (j < J)[None], **kw)[0]


def jack_dial(lp15, b15, prev_mask: int, strength: float):
    """b15 with the jack masks shifted so the row's jack mass is J_R2 + strength (J_copy - J_R2)."""
    b15 = np.asarray(b15, dtype=np.float64)
    if strength >= 1 or prev_mask == 0:
        return b15
    jack = (np.arange(1, 16) & prev_mask) != 0
    lp15 = np.asarray(lp15, dtype=np.float64)
    finite = np.isfinite(lp15)

    def mass(x):
        x = np.where(finite, x, -np.inf)
        p = np.exp(x - x[finite].max())
        return p[jack].sum() / p.sum()

    jp, jq = mass(lp15), mass(lp15 + b15)
    if not (finite & jack).any() or not (finite & ~jack).any():
        return b15
    js = np.clip(jp + strength * (jq - jp), 1e-9, 1 - 1e-9)
    jq = np.clip(jq, 1e-9, 1 - 1e-9)
    out = b15.copy()
    out[jack] += np.log(js / (1 - js)) - np.log(jq / (1 - jq))
    return out


def load_copy(path, map_location='cpu'):
    """(CopyPath in eval mode, checkpoint record) from a ``copytrain`` checkpoint."""
    data = torch.load(path, map_location=map_location, weights_only=False)
    model = CopyPath(CopyConfig(**data['config']))
    model.load_state_dict(data['state'])
    return model.eval(), data


def config_dict(cfg: CopyConfig) -> dict:
    return asdict(cfg)


class CopyBias:
    """Per-chart runtime for ``system.arrange``: ``bias(k, hist, lp15)`` -> [15] for row k.

    ``mel`` is the song's standardised log-Mel on the clock of ``head_ms`` (needed when the path reads audio).
    With ``record`` true, each call appends the row's attention summary to ``self.rows``."""

    def __init__(self, model: CopyPath, head_ms, grid, mel=None, strength: float = 1.0, record: bool = False):
        self.model, self.strength, self.record = model, float(strength), record
        b = grid.beat(np.asarray(head_ms, dtype=np.float64))
        self.b = torch.as_tensor(b, dtype=torch.float32)
        self.keys = torch.as_tensor(timing_keys(b, len(b)))
        self.aud = torch.as_tensor(row_audio(mel, head_ms, grid)) if model.cfg.audio else None
        self.rows = []

    @torch.no_grad()
    def __call__(self, k, hist, lp15):
        lp = torch.as_tensor(np.asarray(lp15), dtype=torch.float32)
        trace = {} if self.record else None
        h = np.asarray(hist[:k], dtype=np.int64) if k else np.zeros(1, np.int64)     # row 0 attends to nothing
        b15 = self.model.chart_bias(torch.tensor([k]), torch.as_tensor(h), self.b, self.keys, self.aud, lp[None],
                                    trace=trace)[0].double().numpy()
        b15 = np.where(np.isfinite(np.asarray(lp15)), b15, 0.0)
        if self.record and k:
            self.rows.append({name: float(v[0]) for name, v in attention_stats(trace).items()})
        return jack_dial(lp15, b15, int(hist[k - 1]) if k else 0, self.strength)

    def hook(self, k, chart, held, logp):
        """The ``head_mask_bias(k, chart, held, logp)`` form of ``r2.sampling.continue_chart`` (zero at EOS)."""
        from ..r2.strain import head_mask_log_probs
        if k >= chart.K:
            return np.zeros(15)
        hist = chart.derived().attack[:k].astype(np.int64) @ (1 << np.arange(4))
        lp15 = head_mask_log_probs(logp.detach().cpu().to(torch.float64), held).numpy()
        return self(k, hist, lp15)


def attention_stats(trace) -> dict:
    """Per query row (B = 1), head-averaged: null mass, entropy (nats), mass on rows whose 9 key entries all
    match, mass at an exact beat lag of 4/8/16/32, attention-weighted sound and context similarity, and whether
    an allowed row with all 9 entries matching exists 4-32 beats back."""
    a = trace['attention'][0].float().cpu()                     # [q, J+1, H]
    J = a.shape[1] - 1
    d, code, allowed = (trace[k][0, :, :J].cpu() for k in ('d', 'code', 'allowed'))
    w = a[:, :J].mean(2)                                         # [q, J]
    full = code == 511
    lag = ((d[..., None] - torch.tensor((4., 8., 16., 32.))).abs() <= .008).any(-1)
    near = (d >= 4 - .008) & (d <= 32 + .008)
    out = dict(null=a[:, J].mean(1).numpy(),
               entropy=(-(a * torch.log(a.clamp_min(1e-12))).sum(1)).mean(1).numpy(),
               key_full=(w * full).sum(1).numpy(), lag=(w * lag).sum(1).numpy(),
               match_available=(full & near & allowed).any(1).numpy())
    if trace['sim'] is not None:
        sim = trace['sim'][0, :, :J].float().cpu()
        z = w.sum(1).clamp_min(1e-12)
        out.update(sound=((w * sim[..., 0]).sum(1) / z).numpy(), context=((w * sim[..., 1]).sum(1) / z).numpy())
    return out


def match_rates(head_beats, tol: int = 1, near=(4.0, 32.0)) -> dict:
    """Shares of rows with an earlier row ``near[0]``-``near[1]`` beats back whose 9 key entries all match:
    exactly, and within ``tol``; and with at least 7 of 9 within ``tol``. Rows only, no model."""
    b = np.asarray(head_beats, dtype=np.float64)
    n = len(b)
    if n < 10:
        return dict(exact=None, tolerant=None, partial7=None)
    keys = torch.as_tensor(timing_keys(b, n))
    lo = np.searchsorted(b, b - near[1] - 1e-6, side='left')
    hi = np.searchsorted(b, b - near[0] + 1e-6, side='right')
    hits = np.zeros((n, 3), bool)
    for k in range(n):
        if hi[k] <= lo[k]:
            continue
        kq = keys[k][None]
        kk = keys[lo[k]:hi[k]]
        hits[k, 0] = bool((key_code(kq, kk, 0) == 511).any())
        tol_code = key_code(kq, kk, tol)
        hits[k, 1] = bool((tol_code == 511).any())
        bits = ((tol_code[:, None] >> torch.arange(9)) & 1).sum(1)
        hits[k, 2] = bool((bits >= 7).any())
    return dict(exact=float(hits[:, 0].mean()), tolerant=float(hits[:, 1].mean()), partial7=float(hits[:, 2].mean()))


def parts_nll(lp, bias, target):
    """Per-row count, lanes-given-count and total NLL of masks under softmax(lp + bias); target in 1..15."""
    logq = torch.log_softmax(lp + bias, -1)
    cols = torch.as_tensor(COUNT[1:] - 1, device=lp.device)
    onehot = nn.functional.one_hot(cols, 4).to(logq.dtype)
    lcount = torch.logsumexp(logq[..., None] + torch.log(onehot.clamp_min(1e-45)), -2)       # [...,4]
    t = (target - 1)[..., None]
    total = -logq.gather(-1, t)[..., 0]
    count = -lcount.gather(-1, cols[t])[..., 0]
    return count, total - count, total
