"""Scoped requests, the request set and its validator (plan v4 section 5.2, amended for q10).

A ``Request`` has a song-time scope [a, b) (b = T means [a, T]), at most one exact target
per property (``ln_share``, ``difficulty`` in absolute star) under the module's nu, the
reserved strength ``'default'``, the default eta (no priority, no transition intervals) and
the reserved ``style`` and ``demand`` fields. Every reserved field set away from its default
raises ``ContractError``: R2 cannot honour it, so it is never accepted and ignored.

``RequestSet`` keeps each request with the committed boundary g_u at which it was added
(``None`` is 0^-, the start of generation) across continuation calls. A request is valid
only if g_u < a. Before its start (g < a) a request may be withdrawn or replaced; once the
committed boundary reaches a it can be neither cancelled nor changed. Two targets for the
same property with intersecting scopes make the set invalid; targets for different
properties all apply and are recorded as co-active. ``effective_track`` turns the set into
the per-kind ``Interval`` track the model reads (one interval per target, never merged);
rule L (``locality.py``) then decides which decisions read each interval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from .common import ContractError
from .features import Interval
from .properties import MIN_STAR_MS, NU_HASH, NU_ID, row_range

KIND_INDEX = {'ln_share': 0, 'difficulty': 1}
KIND_NAME = {v: k for k, v in KIND_INDEX.items()}
RESIDUAL_FLAG_STAR = 1.5
LN_TRAINED_BEATS = (8.0, 512.0)
STAR_TRAINED_MS = (30_000.0, 60_000.0, 120_000.0)


@dataclass(frozen=True)
class Eta:
    priority: object = None
    transitions: tuple = ()


@dataclass(frozen=True)
class Target:
    value: float
    nu: str | None = None          # None: the module's nu for the property
    strength: str = 'default'


@dataclass(frozen=True)
class Request:
    id: str
    scope: tuple                   # (a, b) in song ms
    property: dict = field(default_factory=dict)   # {'ln_share' | 'difficulty': Target}
    style: object = None
    eta: Eta = Eta()
    demand: object = None

    def to_json(self):
        return dict(id=self.id, scope=list(self.scope), style=self.style, demand=self.demand,
                    eta=dict(priority=self.eta.priority, transitions=list(self.eta.transitions)),
                    property={k: asdict(t) for k, t in self.property.items()})

    @classmethod
    def from_json(cls, data):
        eta = data.get('eta') or {}
        return cls(str(data['id']), tuple(float(x) for x in data['scope']),
                   {k: Target(float(t['value']), t.get('nu'), t.get('strength', 'default'))
                    for k, t in (data.get('property') or {}).items()},
                   data.get('style'), Eta(eta.get('priority'), tuple(eta.get('transitions') or ())),
                   data.get('demand'))


@dataclass
class Entry:
    request: Request
    added_at: float | None         # g_u; None is 0^-
    withdrawn_at: float | None = None
    withdrawn: bool = False

    def to_json(self):
        return dict(request=self.request.to_json(), added_at='0^-' if self.added_at is None else self.added_at,
                    withdrawn=self.withdrawn, withdrawn_at=self.withdrawn_at)


def _g(frontier):
    return -math.inf if frontier is None else float(frontier)


def check_schema(request: Request, song_ms: float):
    """Step 1 of validation: directives, reserved fields, nu, scope bounds, value ranges."""
    if not isinstance(request.id, str) or not request.id:
        raise ContractError('A request needs a non-empty string id')
    if request.style is not None:
        raise ContractError('Style directives are not built in R2 (stage 4); the style field must be None')
    if request.demand is not None:
        raise ContractError('Gameplay-demand requests have no interface; the demand field must be None')
    if not isinstance(request.eta, Eta):
        raise ContractError('eta must be an Eta')
    if request.eta.priority is not None:
        raise ContractError('Priority is not built in R2: it has no mechanism to follow one directive first')
    if tuple(request.eta.transitions):
        raise ContractError('Transition intervals (lead-in, release) are not built; only the default eta is')
    if not request.property:
        raise ContractError('A request carries at least one directive')
    a, b = (float(x) for x in request.scope)
    if not (math.isfinite(a) and math.isfinite(b) and 0.0 <= a < b <= float(song_ms)):
        raise ContractError(f'Scope {request.scope} is not 0 <= a < b <= T = {song_ms}')
    for kind, target in request.property.items():
        if kind not in KIND_INDEX:
            raise ContractError(f'Unknown property {kind!r}; R2 supports {sorted(KIND_INDEX)}')
        if not isinstance(target, Target):
            raise ContractError('A property directive is a Target with one value (no ranges)')
        if target.nu is not None and target.nu != NU_ID[kind]:
            raise ContractError(f'Unsupported nu {target.nu!r} for {kind}; R2 supports {NU_ID[kind]!r}')
        if target.strength != 'default':
            raise ContractError('strength levels above the default are not built')
        v = target.value
        if isinstance(v, (list, tuple)) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ContractError('A target is one finite value, not a range')
        if kind == 'ln_share' and not 0.0 <= float(v) <= 1.0:
            raise ContractError('An LN-share target lies in [0, 1]')
        if kind == 'difficulty' and b - a < MIN_STAR_MS:
            raise ContractError('A difficulty scope needs at least 30 s: nu is undefined below')


def _intersect(x: Request, y: Request):
    return x.scope[0] < y.scope[1] and y.scope[0] < x.scope[1]


class RequestSet:
    """The requests in effect for a chart, kept across continuation calls."""

    def __init__(self, song_ms: float, entries=()):
        self.song_ms = float(song_ms)
        self.entries: list[Entry] = list(entries)

    def active(self):
        return [e for e in self.entries if not e.withdrawn]

    def get(self, request_id):
        for e in self.active():
            if e.request.id == request_id:
                return e
        raise ContractError(f'No active request {request_id!r}')

    def add(self, request: Request, frontier=None) -> 'RequestSet':
        """Add ``request`` at the committed boundary ``frontier`` (None: start of generation)."""
        check_schema(request, self.song_ms)
        if any(e.request.id == request.id for e in self.entries):
            raise ContractError(f'Request id {request.id!r} is already used in this set')
        if not _g(frontier) < request.scope[0]:
            raise ContractError('request added at or after its start')
        for e in self.active():
            if _intersect(e.request, request):
                same = sorted(set(e.request.property) & set(request.property))
                if same:
                    raise ContractError(f'same-quantity overlap: {e.request.id!r} and {request.id!r} on {same}')
        self.entries.append(Entry(request, None if frontier is None else float(frontier)))
        return self

    def withdraw(self, request_id, frontier=None) -> 'RequestSet':
        e = self.get(request_id)
        if not _g(frontier) < e.request.scope[0]:
            raise ContractError('a request can be neither cancelled nor changed once its scope has started')
        e.withdrawn, e.withdrawn_at = True, None if frontier is None else float(frontier)
        return self

    def replace(self, request_id, request: Request, frontier=None) -> 'RequestSet':
        e = self.get(request_id)
        if not _g(frontier) < e.request.scope[0]:
            raise ContractError('a request can be neither cancelled nor changed once its scope has started')
        self.withdraw(request_id, frontier)
        try:
            return self.add(request, frontier)
        except ContractError:
            e.withdrawn, e.withdrawn_at = False, None
            raise

    def check_frontier(self, frontier):
        """A continuation call keeps every request; it may not resume from before a request's addition."""
        for e in self.active():
            if e.added_at is not None and _g(frontier) < e.added_at:
                raise ContractError(f'Frontier {frontier} precedes the addition boundary of {e.request.id!r}')

    def copy(self):
        return RequestSet(self.song_ms, [Entry(e.request, e.added_at, e.withdrawn_at, e.withdrawn)
                                         for e in self.entries])

    def to_json(self):
        return dict(song_ms=self.song_ms, nu_hash=NU_HASH, entries=[e.to_json() for e in self.entries])

    @classmethod
    def from_json(cls, data):
        out = cls(float(data['song_ms']))
        for e in data['entries']:
            g = e['added_at']
            out.entries.append(Entry(Request.from_json(e['request']), None if g == '0^-' else float(g),
                                     e.get('withdrawn_at'), bool(e.get('withdrawn'))))
        return out


@dataclass
class Effective:
    track: tuple                   # Interval per target, as the model reads it
    provenance: list               # per interval: dict(request, kind, target, b_s, v_res)
    flags: list                    # dict(request, kind, flag)
    co_active: list                # pairs of (request, kind) with intersecting scopes on different kinds


def scope_beats(grid, a, b):
    return float(grid.beat(b) - grid.beat(a))


def effective_track(requests: RequestSet, head_ms, grid, *, star_value: str = 'absolute', baseline=None) -> Effective:
    """Steps 4-6 of validation: the effective track, provenance, feasibility flags and co-active pairs."""
    T = requests.song_ms
    track, prov, flags, pairs = [], [], [], []
    active = requests.active()
    for e in active:
        r = e.request
        a, b = r.scope
        lo, hi = row_range(head_ms, a, b, T)
        whole = a == 0.0 and b >= T
        for kind, target in sorted(r.property.items()):
            value, b_s, v_res = float(target.value), None, None
            if hi == lo:
                flags.append(dict(request=r.id, kind=kind, flag='ungovernable: no head row in the scope'))
                if kind == 'ln_share':
                    flags.append(dict(request=r.id, kind=kind, flag='unattainable: the readout will be undefined'))
            if kind == 'ln_share':
                beats = scope_beats(grid, a, b)
                if not (whole or LN_TRAINED_BEATS[0] <= beats <= LN_TRAINED_BEATS[1]):
                    flags.append(dict(request=r.id, kind=kind, flag=f'extrapolation: {beats:.2f} beats'))
            else:
                if not (whole or any(abs((b - a) - L) <= 1.0 for L in STAR_TRAINED_MS)):
                    flags.append(dict(request=r.id, kind=kind, flag=f'extrapolation: {(b - a) / 1000:.3f} s'))
                if star_value == 'residual':
                    if baseline is None:
                        raise ContractError('Residual difficulty conditioning needs the frozen baseline b(S)')
                    b_s = float(baseline.predict(head_ms, a, b))
                    v_res = value - b_s
                    if abs(v_res) > RESIDUAL_FLAG_STAR:
                        flags.append(dict(request=r.id, kind=kind,
                                          flag=f'outside trained residual range: {v_res:+.3f} star (clipped)'))
                    value = v_res
                elif star_value != 'absolute':
                    raise ContractError(f'Unknown star_value {star_value!r}')
            track.append(Interval(KIND_INDEX[kind], float(a), float(b), value))
            prov.append(dict(request=r.id, kind=kind, a=float(a), b=float(b), target=float(target.value),
                             nu=NU_ID[kind], strength=target.strength, b_s=b_s, v_res=v_res))
    for i, x in enumerate(active):
        for y in active[i:]:
            if x is not y and not _intersect(x.request, y.request):
                continue
            for kx in x.request.property:
                for ky in y.request.property:
                    if kx != ky and (x is not y or kx < ky):
                        pairs.append(dict(a=dict(request=x.request.id, kind=kx), b=dict(request=y.request.id, kind=ky)))
    from .conditions import validate_track
    validate_track(tuple(track), song_ms=T, star_value=star_value)
    return Effective(tuple(track), prov, flags, pairs)


def from_track(track, song_ms: float, *, star_value='absolute', baseline=None, head_ms=None) -> RequestSet:
    """The request set a training track stands for: every interval added at 0^- (section 5.2, training)."""
    out = RequestSet(song_ms)
    for i, iv in enumerate(sorted(track, key=lambda iv: (iv.kind, iv.a))):
        kind = KIND_NAME[iv.kind]
        value = iv.value
        if kind == 'difficulty' and star_value == 'residual':
            value = float(baseline.predict(head_ms, iv.a, iv.b)) + iv.value
        out.add(Request(f'{kind}-{i}', (iv.a, min(iv.b, song_ms)), {kind: Target(float(value))}), None)
    return out
