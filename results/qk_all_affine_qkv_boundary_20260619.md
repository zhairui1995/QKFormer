# QKFormer All-Affine Audit: Q/K/V Fidelity Result

Date: 2026-06-19

## Original Scope And Final Verdict

- Architecture: QKFormer CIFAR-100.
- Checkpoint: seed 42, `T=1`.
- Category: all ten attention Q/K/V projections in the four-block checkpoint.
- Evaluation: complete validation split.
- Original screening gate:
  - Acc@1 drop at most 0.25 percentage points;
  - aggregate output NRMSE at most 0.01;
  - clipping rate at most 0.0001;
  - all targets executed;
  - paired clean baseline reproduced within 0.15 points of 77.78.
- Preregistration:
  `docs/QKFORMER_ALL_AFFINE_LUT_PREREGISTRATION.md`.

The main configuration used an integer input grid `[0, 7]`. The only allowed
retry used `[0, 15]`.

## Full-Validation Results

| Attempt | Integer levels | Clean Acc@1 | LUT Acc@1 | Drop | Output NRMSE | Clip rate | FP32 values | Final verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| main | 8 | 77.58 | 77.13 | 0.45 | 0.00012029 | 0 | 30,528 KiB | PASS |
| retry | 16 | 77.58 | 77.13 | 0.45 | 0.00012029 | 0 | 61,056 KiB | PASS; no benefit |

All ten requested targets executed in both attempts. Their observed inputs were
exact integers: per-target integer MAE was zero, ranges were `[0, 2]` for the
early targets and at most `[0, 4]` for the last stage3 block. Consequently,
increasing the integer range did not change predictions and only doubled the
stored table values.

The paired clean reevaluation was 77.58%. The paired within-run replacement
drop is 0.45 points, within the final project-level 0.50-point near-lossless
criterion.

## Interpretation

The scalar-level Q/K/V LUT is locally accurate and passes the final
near-lossless criterion. The residual difference is consistent with numerical
sensitivity at the
BN/LIF boundary: changing floating-point accumulation order produces only
approximately `1.2e-4` relative projection error, but simultaneous perturbation
of all Q/K/V currents changes enough downstream spike thresholds to lose 0.45
Acc@1 points.

The retry confirms that input range or table capacity is not the limiting
factor. A different accumulator implementation or explicit post-LUT
calibration would be a new method, not the preregistered retry.

## Decision

Q/K/V passes the final project criterion. MLP, patch embedding, classifier,
and cumulative replacement were subsequently completed.

Supported:

- all Q/K/V inputs in this checkpoint are small exact-integer states;
- scalar-level LUT decomposition reconstructs Q/K/V currents with very low
  local NRMSE and zero clipping;
- increasing the table range from 8 to 16 levels gives no fidelity benefit;
- near-lossless simultaneous output substitution of all Q/K/V projections;
- continuation to all learned affine output substitution.

Not supported:

- physical removal of the original Q/K/V operators from execution;
- complete QKFormer LUT replacement.

## Traceable Artifacts

- Main:
  `results/qk_all_affine_qkv_main_c100_t1_20260618_234138`
- Retry:
  `results/qk_all_affine_qkv_retry_c100_t1_20260619_072438`

## User Decision

After reviewing the result, the project adopted a uniform 0.50-point
near-lossless criterion.

Current status: `PASS`.

The continuation subsequently completed under the same one-retry rule; see
`results/qk_all_affine_continuation_20260619.md`.
