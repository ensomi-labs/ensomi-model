"""Bar-only intervention, ported from the supplied design scratch variant.

The sampled-beat likelihood ratio uses pi=0.20; original meter weights stay
fixed. No package decoder is modified. See prereg.md for its causal scope.
"""
import numpy as np

def bars_v(d, cfg, prior_mode='bar', pi=None, restart=None):
    n = len(d)
    meters = np.array(cfg.meters); prior = np.array(cfg.meter_logprior, float); M = len(meters)
    offs = np.concatenate([[0], np.cumsum(meters)[:-1]]); S = int(meters.sum())
    m_of = np.repeat(np.arange(M), meters); k_of = np.concatenate([np.arange(m) for m in meters])
    first, last = offs, offs + meters - 1; notfirst = np.flatnonzero(k_of > 0)
    dc = np.clip(d, cfg.eps, 1 - cfg.eps)
    e_down, e_up = np.log(dc), np.log1p(-dc)
    if pi is not None:
        e_down, e_up = e_down - np.log(pi), e_up - np.log1p(-pi)
    if prior_mode == 'bar':
        C = np.where(np.eye(M, dtype=bool), 0.0, -cfg.meter_change_penalty); per_bar = prior
    else:   # a meter run pays its prior once, at its start
        C = np.where(np.eye(M, dtype=bool), 0.0, -cfg.meter_change_penalty + prior[None, :]); per_bar = np.zeros(M)
    if restart is None:
        restart = np.zeros(n, bool)
    v = np.where(k_of == 0, e_down[0], e_up[0]) + (per_bar[m_of] if prior_mode == 'bar' else prior[m_of])
    back = np.zeros((n, M), dtype=np.int64)
    for i in range(1, n):
        new = np.empty(S); new[notfirst] = v[notfirst - 1]
        E = v[last][:, None] + C; src_e = E.argmax(0); best_e = E[src_e, np.arange(M)]
        pen = cfg.bar_restart_penalty if restart[i] else cfg.bar_restart_anywhere
        rmax = np.array([v[offs[mi]:offs[mi] + meters[mi]].max() for mi in range(M)])
        rarg = np.array([offs[mi] + int(v[offs[mi]:offs[mi] + meters[mi]].argmax()) for mi in range(M)])
        R = rmax[:, None] + C - pen; src_r = R.argmax(0); best_r = R[src_r, np.arange(M)]
        use_r = best_r > best_e
        new[first] = np.where(use_r, best_r, best_e) + per_bar
        back[i] = np.where(use_r, rarg[src_r], last[src_e])
        new[first] += e_down[i]; new[notfirst] += e_up[i]; v = new
    s = int(np.argmax(v)); states = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        states[i] = s
        if i == 0: break
        s = int(back[i, m_of[s]]) if k_of[s] == 0 else s - 1
    return k_of[states] == 0, meters[m_of[states]]

def case(meter_true, n_bars, d_hi, d_lo, marks=(), lead=(0.9, 0.05), nlead=4):
    td, sec = [], []
    for meter, nb, insec in ((4, nlead, False), (meter_true, n_bars, True), (4, nlead, False)):
        for b in range(nb):
            for k in range(meter):
                td.append(k == 0); sec.append(insec)
    td, sec = np.array(td), np.array(sec)
    d = np.where(td, np.where(sec, d_hi, lead[0]), np.where(sec, d_lo, lead[1])).astype(float)
    for m in marks: d[m] = 0.999
    return d, td, sec

def score(dm, td, sec):
    rec = (dm & td & sec).sum() / max((td & sec).sum(), 1); prec = (dm & td & sec).sum() / max((dm & sec).sum(), 1)
    return rec, prec, int(((dm != td) & ~sec).sum())
