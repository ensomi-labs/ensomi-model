"""Training queries for independently optional style and numeric requests."""
from dataclasses import dataclass

from ..bounded_typed_continuation.features import TIME_DIM
from ..typed_audio_continuation.controls import PER_FIELD_SCOPE, SHARED_SCOPE


@dataclass(frozen=True)
class StyleWithoutLn:
    """Hide LN request only where a style field is actually observed.

    This is a neural-condition training view, not an executable control
    schedule. It intentionally exposes no spans for quota/amount feedback.
    Other numeric/style values, availability and their scope clocks stay
    unchanged. Unannotated portions do not become unconditional examples
    from a style-balanced population. Source rows/targets never change.
    """
    source: object

    @property
    def style_names(self):
        return self.source.style_names

    def at(self, times, *, encoding=SHARED_SCOPE):
        values = self.source.at(times,encoding=encoding)
        n = 2+len(self.style_names)
        active = values[:,n+2:2*n].any(-1)
        values[active,1] = 0
        values[active,n+1] = 0
        if encoding == PER_FIELD_SCOPE:
            values[active,2*n+2*TIME_DIM:2*n+4*TIME_DIM] = 0
        return values


def style_ln_views(controls, start_ms, end_ms):
    """Return equal-weight source/control views for the scored real-time range.

    A second view exists only when an observed style and known LN amount
    overlap the scored interval. Its mask is applied at each query, including
    release/wait queries. Existing known-amount supervision is retained.
    Both views must use the same source example, audio encoding and weights;
    averaging their losses does not double that source's sampling mass.
    """
    applicable = any(span.ln_fraction is not None and bool(span.style)
                     for span in controls.resolved_ranges(start_ms,end_ms))
    return (controls,StyleWithoutLn(controls)) if applicable else (controls,)
