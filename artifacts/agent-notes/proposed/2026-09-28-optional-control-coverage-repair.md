# Agent Note: Style-known, LN-unknown training coverage

Note ID: 2026-09-28-optional-control-coverage-repair
Status: proposed
Kind: investigation
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 1658123e8a43f7881a9a5175c841c811d04edd97
Scope: Independent control missingness and conditional arrangement learning
Related: 2026-09-28-operational-response-frontier, 2026-09-28-clean-joint-proposal-learning

## Verified recipe gap

The user asks why ordinary low-difficulty arrangements still fail and demands
skeptical architecture/recipe review. Continuing pressure-frontier work exposed
an independent data issue in prepare-v2.py: drop_ln and drop_stars are enabled
only for branch==natural. The human/style branch always exposes numeric LN
amount. This is not the independently optional request distribution used by
the actual D4+Stream/no-LN-control generation request.

The frozen8192example source ledger(a48c9cc55f60fd6353295513f060ac23fd7f5487cf798907eb3eca8b6d716862)
has221draws/21charts/235effective pieces with prominent Stream. All retain LN
condition. AtD3.5–4.5,68draws/10charts/74pieces, again0LN-hidden pieces.
Actual LN fraction median0; only2/74pieces have>=.5LN. Thus an LN-dominated
training target is not a sufficient account of that conditional generation.
Exposure counts are not independent samples. Earlier source inheritance may
have other exposures; this census does not prove the entire historical model
never saw that condition combination.

The exploratory census resolves actual control boundaries, loads actual row
arrays and excludes style annotations outside the scored range. It is saved
under action-response-frontier-v1/stream-source-audit-exploratory.json.
The initial string search for style key Stream found none; the actual key is
stream-organization, with prominent1/supporting0/absent-1. This key discovery
preceded the actual reported census.

## Implemented query view

1658123adds conditional_views.StyleWithoutLn/style_ln_views. Same source,
audio, state, support and weight; paired known-amount and hidden-amount
conditions averaged equally. Hide value/availability/per-field clocks only
where an effective style is observed. Unannotated portions retain original
amount information, protecting the natural unconditional prior from
balanced-human exposure. Supporting/absent style labels are still observed.

The query view is deliberately not a control schedule and has no spans.
Sampling replay skips amount-episode reconstruction when feedback is disabled;
it then needs only at()/style_names. This prevents accidentally reintroducing
the hidden source amount through a quota controller.7focused conditional-view
and sampling checks pass1.94s. Existing R1/H/R weights are unchanged.

## Experiment Card: optional-control-response-v1

Revision: 1
Accepted revision: none
Execution authority: explicit user instruction to repair recipes and models,
plus the active full-system goal. Exploratory, no adoption claim.

Question: does the existing row law change toward excess LN when the source
amount is hidden but actual style/audio/history remain fixed? This is a direct
conditional-response diagnostic, not proof of native quality or a percentage
attribution of collapse.

Parent is joint-release80 checkpoint
b57934728a77abfef0d4bd1ea6387d7cb6f8582c6e450bdd890bfeb1d380ebd6.
Use all14distinct source/intervals among effective prominentStream D3.5–4.5
in the frozen ledger; no selection from generated outputs. Each actual
constant-control scope reports separately. Other source times/controls are
not pooled into it. Both views share complete audio and factual row/H history;
compare expected LNheads/heads in the scored row law and actual source counts.
Full source future H is teacher forcing, not an audio-only native result.

CPU2threads,180s bound, fresh
artifacts/joint-audio/20260928-optional-control-repair-v1/probe-v1,
no overwrite/restart. Source1658123 and script/data/checkpoint bytes pinned.
probe-plan.json was superseded before execution by probe-v2-plan.json to
retain separate control-range reports. Run probe_v2.py, not the older script.
Stop on source/hash drift, runtime bound or malformed source probability.

If conditional odds support the concern, compare matched learning on the same
source slice with versus without this paired view. Do not simply hide all
controls on balanced data, force Stream LNratio0 at inference, or promote from
teacher-state odds/NLL. Native pressure, amount, ordinary role organization,
rhythm and runtime remain independent required checks.

## Probe result

Session59289completed exit0 in5.458s:14distinct source/intervals,15separate
effective control ranges. ResultSHA
70aad40f93e8958f1601de649ff24fc0c61f3e731c4024276145cb0ce88258b6.
Hidden-minus-known expected LN-head fraction median.0025786, max.0253596,
min-.0044471. Factual TAP Stream states still predict predominantly TAP.
This supports a real coverage defect but rejects it as a sufficient account
of the huge native LN bias. No matched NN fine-tune was launched on a false
primary-cause claim. Query-view support stays available for the next sound
recipe; native-prefix/timing/proposal corrections now have higher priority.
