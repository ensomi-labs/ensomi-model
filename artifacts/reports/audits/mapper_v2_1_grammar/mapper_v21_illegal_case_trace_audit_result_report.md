# Mapper v2.1 Illegal-Case Trace Audit Result Report

## Scope

This pass executes `mapper_v21_illegal_case_trace_audit_experiment_card.md`. It reruns selected v2.1 fixed-slice cases with a default-off logits trace transform. It does not change mapper defaults, tokenizer, grammar, replay, model weights, or training.

## Decision

Decision: `TEST_NEXT`.

- Reason: primary illegal cases reproduced and received concrete trace classifications
- Runs: `4`
- Illegal runs: `2`
- Primary illegal reproduced: `2` / `2`
- Classified illegal runs: `2`
- Classification families: `{'anti_rigid_path_damage': 1, 'legal_contrast': 2, 'zero_valid_terminal_state': 1}`
- Anti-rigid blocked count: `20`
- Changed trace steps: `20`

## Case Table

| Case | Mode | Legal | Expected legal | Terminal ms | Tokens | Timepoints | Blocks | Family | Top invalid reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | `baseline` | `False` | `False` | `7990` | `64` | `12` | `0` | `zero_valid_terminal_state` | `TIME_SHIFT moves past target_end_ms` |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | `guard` | `True` | `True` | `16000` | `199` | `54` | `19` | `legal_contrast` | `TIME_SHIFT moves past target_end_ms` |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | `baseline` | `True` | `True` | `16000` | `275` | `74` | `0` | `legal_contrast` | `TIME_SHIFT moves past target_end_ms` |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | `guard` | `False` | `False` | `7990` | `98` | `32` | `1` | `anti_rigid_path_damage` | `TIME_SHIFT moves past target_end_ms` |

## Illegal Trace Details

### `04_oomori_seiko_justadice_tv_size_remu_normal` / `baseline`

- Family: `zero_valid_terminal_state`
- Terminal valid tokens: `0`
- Terminal matches carry-out: `False`
- Terminal state current_ms: `7990`
- Terminal open_mask: `[False, False, True, True]`
- Terminal open_start_ms: `[None, None, 7990, 7990]`
- Terminal emitted_lane_mask: `[True, False, True, True]`
- Terminal invalid reasons: `{'TIME_SHIFT moves past target_end_ms': 21, 'same-time lane-action ordering/duplicate violation': 12, 'EOS not legal in this window': 1, 'TIME_SHIFT to write_end_ms requires resulting state to equal ln_carry_out': 1}`
- Anti-rigid examples: `[]`

### `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` / `guard`

- Family: `anti_rigid_path_damage`
- Terminal valid tokens: `0`
- Terminal matches carry-out: `False`
- Terminal state current_ms: `7990`
- Terminal open_mask: `[False, True, False, False]`
- Terminal open_start_ms: `[None, 7990, None, None]`
- Terminal emitted_lane_mask: `[True, True, False, True]`
- Terminal invalid reasons: `{'TIME_SHIFT moves past target_end_ms': 21, 'same-time lane-action ordering/duplicate violation': 12, 'EOS not legal in this window': 1, 'TIME_SHIFT to write_end_ms requires resulting state to equal ln_carry_out': 1}`
- Anti-rigid examples: `[{'token_id': 14, 'token_name': 'TS_300', 'spacing_ms': 320, 'first_piece_ms': 300, 'current_ms': 1600, 'token_index': 15, 'required_alternative': True, 'tap_only_run': False, 'mode': 'hard_block'}]`


## What Passed

- The audit used the existing runtime rollout path and recorded generated-prefix trace state.
- Baseline mode uses an identity trace transform; guard mode delegates to the existing anti-rigid transform.
- Per-case JSON artifacts include tail valid-mask/logit records and terminal invalid-reason counts.

## Interpretation

The trace audit reproduced the target illegal cases and converted the aggregate legality failure into concrete failure families. The next loop can be a single legality-first grammar repair card.

## Next Step

Create a legality-first v2.1 grammar repair card targeted to the observed failure family.
