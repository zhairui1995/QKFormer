# Native affine + BN-folded + finite LIF transition LUT audit

Date: 2026-06-19

## Registered gate

- Dataset/checkpoint scope: QKFormer CIFAR-100, seed 42, full validation.
- Temporal settings: `T=1` and `T=4`.
- Accuracy gate: paired Acc@1 drop `<= 1.50 pp`.
- LIF clipping gate: aggregate transition-LUT clipping rate `<= 1e-4`.
- Execution gate: all 32 Conv1d/Conv2d/Linear targets and all 35 LIF targets
  must execute through replacements; original target modules, paired BN
  modules, and original SpikingJelly LIF modules must have zero evaluation
  calls.

## Final method

Every frozen Conv1d/Conv2d/Linear module is replaced by a scalar-level
lookup-and-accumulate module. Paired inference BatchNorm parameters are folded
into the corresponding affine LUT and the BN modules are removed. Every
MultiStepLIFNode is replaced by an 8-bit finite per-element transition table
mapping `(membrane state, input)` to `(spike, next membrane state)`.

Calibration identified two attention LIF inputs as exact integers. Preserving
these inputs as integer LUT addresses, instead of decoding them through a
uniform floating-point grid, removes artificial nonzero membrane states and
the resulting recurrent clipping.

## Full-validation results

| Setting | Clean Acc@1 | Replacement Acc@1 | Drop | LIF clip rate | LIF FP32 values | Gate |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-100 `T=1` | 77.58 | 77.38 | 0.20 pp | 2.45e-9 | 16,976 KiB | PASS |
| CIFAR-100 `T=4` | 81.02 | 80.80 | 0.22 pp | 1.67e-9 | 16,980 KiB | PASS |

Both settings execute all 32 affine targets and all 35 LIF targets. Remaining
original affine, paired BN, and original LIF module counts and evaluation-call
counts are zero. The final T=4 clipping rate is approximately 59,800 times
below the registered `1e-4` threshold.

Machine-readable evidence:

- `results/qk_native_affine_all_affine_main_c100_t1_20260619_185312/metrics.json`
- `results/qk_native_affine_all_affine_main_c100_t1_20260619_185312/summary.csv`
- `results/qk_native_affine_all_affine_main_c100_t4_20260619_183707/metrics.json`
- `results/qk_native_affine_all_affine_main_c100_t4_20260619_183707/summary.csv`

## Claim boundary

This audit supports the claim that all learned affine operators, paired
inference BatchNorm operations, and all LIF state transitions in the evaluated
QKFormer can be removed from the runtime graph and replaced by LUT-native
operators while staying within the 1.50-pp full-validation gate at both `T=1`
and `T=4`.

It is not yet a complete QKFormer LUT replacement. Pooling, residual addition,
SSA matrix products/gating, and global/spatial/temporal reductions remain
ordinary arithmetic. The table sizes are fidelity evidence, not storage,
latency, energy, or hardware-acceleration evidence.
