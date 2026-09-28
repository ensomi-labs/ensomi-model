"""Composable evidence reports with stable scope identities and no pooled verdict."""
from .alignment import MelDescriptors, audio_correspondence
from .hold_relations import hold_relations
from .hold_interactions import hold_interactions
from .head_recurrence import head_recurrence
from .tap_organization import tap_organization


FEATURES = ('H_Hz','column0_attack_Hz','column1_attack_Hz','column2_attack_Hz','column3_attack_Hz',
            'column0_held_fraction','column1_held_fraction','column2_held_fraction','column3_held_fraction',
            'LN_head_Hz','release_Hz')


def evaluate_scopes(trace,scopes,*,mel=None,identity=None):
    """Return independent reports; optional identity records caller-verified bytes.

    This report contains measurements and witnesses, not a single quality score.
    Source/model/audio/control provenance belongs in identity. Sustained envelope
    comparisons can be attached from player_response without redefining its cost.
    """
    descriptor=MelDescriptors(mel) if mel is not None else None
    reports=[]
    for scope in scopes:
        row=trace.scope_report(scope)
        row['LN_timing_relations']=hold_relations(trace,scope)
        row['LN_interactions']=hold_interactions(trace,scope)
        row['head_recurrence']=head_recurrence(trace,scope)
        row['TAP_organization']=tap_organization(trace,scope)
        row['contrasts']=trace.contrasts(scope)
        row['audio_correspondence']=(None if descriptor is None else audio_correspondence(trace,descriptor,scope))
        reports.append(row)
    return dict(format='gameplay-temporal-evaluation/v1',coverage_ms=trace.coverage_ms,
        feature_order=list(FEATURES),identity=identity,scopes=reports,
        interpretation='Temporal evidence; no semantic style, physiological or complete playability verdict.')
