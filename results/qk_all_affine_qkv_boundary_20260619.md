# QKFormer All-Affine Audit: Q/K/V Boundary Result

Date: 2026-06-19

## Preregistered Scope

- Architecture: QKFormer CIFAR-100.
- Checkpoint: seed 42, `T=1`.
- Category: all ten attention Q/K/V projections in the four-block checkpoint.
- Evaluation: complete validation split.
- Gate:
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

| Attempt | Integer levels | Clean Acc@1 | LUT Acc@1 | Drop | Output NRMSE | Clip rate | FP32 values | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| main | 8 | 77.58 | 77.13 | 0.45 | 0.00012029 | 0 | 30,528 KiB | FAIL |
| retry | 16 | 77.58 | 77.13 | 0.45 | 0.00012029 | 0 | 61,056 KiB | FAIL |

All ten requested targets executed in both attempts. Their observed inputs were
exact integers: per-target integer MAE was zero, ranges were `[0, 2]` for the
early targets and at most `[0, 4]` for the last stage3 block. Consequently,
increasing the integer range did not change predictions and only doubled the
stored table values.

The paired clean reevaluation was 77.58%, 0.20 points below the registered
77.78% hook baseline and therefore also missed the strict baseline-reproduction
gate. The primary failure is independent of that discrepancy: the paired
within-run replacement drop is 0.45 points, above the 0.25-point gate.

## Interpretation

The scalar-level Q/K/V LUT is locally very accurate but not end-to-end
lossless. The working mechanism hypothesis is numerical sensitivity at the
BN/LIF boundary: changing floating-point accumulation order produces only
approximately `1.2e-4` relative projection error, but simultaneous perturbation
of all Q/K/V currents changes enough downstream spike thresholds to lose 0.45
Acc@1 points.

The retry confirms that input range or table capacity is not the limiting
factor. A different accumulator implementation or explicit post-LUT
calibration would be a new method, not the preregistered retry.

## Stop Decision

Per the user-approved stop rule and preregistration, the all-affine experiment
sequence stops here. MLP, patch embedding, classifier, and cumulative
`all_affine` replacements were not launched.

Supported:

- all Q/K/V inputs in this checkpoint are small exact-integer states;
- scalar-level LUT decomposition reconstructs Q/K/V currents with very low
  local NRMSE and zero clipping;
- increasing the table range from 8 to 16 levels gives no fidelity benefit.

Not supported:

- near-lossless simultaneous replacement of all Q/K/V projections;
- all learned affine operator replacement;
- complete QKFormer LUT replacement.

## Traceable Artifacts

- Main:
  `results/qk_all_affine_qkv_main_c100_t1_20260618_234138`
- Retry:
  `results/qk_all_affine_qkv_retry_c100_t1_20260619_072438`
