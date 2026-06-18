# QKFormer All-Affine LUT Replacement Preregistration

Date: 2026-06-18

## Scope And Claim Boundary

This experiment audits the remaining learned affine operators in the
QKFormer CIFAR-100 seed-42 checkpoint:

1. all attention Q/K/V projections;
2. all MLP `mlp1_conv` and `mlp2_conv` projections;
3. all patch-embedding convolutions;
4. the classifier;
5. the already studied attention `proj_conv` projections in the final
   cumulative row.

The replacement is a scalar-level LUT decomposition of each frozen convolution
or linear layer. Each quantized input scalar indexes a vector of precomputed
output-channel contributions, and the vectors are accumulated with the frozen
bias.

This is an **all learned affine operator** audit. It is not complete QKFormer
replacement. Batch normalization, LIF state dynamics, pooling, residual
addition, and the SSA `K^T V` / `Q(...)` matrix products remain unchanged.
The hook also executes the original operator before substituting the LUT
result, so the experiment validates numerical and classification fidelity, not
measured runtime removal.

## Fixed Experimental Setting

- Architecture: QKFormer.
- Dataset: CIFAR-100.
- Training checkpoint: seed 42.
- Main time step: `T=1`.
- Evaluation: complete validation split.
- Calibration: 32 fixed training batches, seed 42.
- No training or weight update.
- Categories are tested in this order:
  1. `qkv`
  2. `mlp`
  3. `patch`
  4. `classifier`
  5. `all_affine`, only if all four individual categories pass.

The experiment stops when one category fails both its main and retry
configuration. Later categories are not run after that stop.

## Quantization And LUT Configurations

Residual/spike-path inputs are audited against an integer grid.

- Main configuration: round and clip to `[0, 7]` (8 levels).
- Only retry: round and clip to `[0, 15]` (16 levels).

Continuous inputs are affine-quantized using calibration-set minimum and
maximum.

- Main configuration: 8 bits.
- Only retry: 10 bits.

`patch_embed1.proj_conv` and the classifier use continuous quantization.
Other targets use the integer grid unless the audit reports otherwise. A
failed main configuration may only be retried with the paired configuration
above; no additional bit-width, clipping, stage, checkpoint, or seed search is
allowed.

## Gates

A category passes only if all conditions hold:

1. paired clean baseline is reproduced;
2. full-validation Acc@1 drop is at most `0.25` percentage points;
3. aggregate operator-output NRMSE is at most `0.01`;
4. validation clipping rate is at most `0.0001`;
5. all requested targets executed and produced finite metrics.

The cumulative `all_affine` row uses the same gates.

## Interpretation

If all individual categories and `all_affine` pass, the supported claim is:

> All learned convolutional and linear operators in the evaluated QKFormer
> configuration admit the reported scalar-level LUT-equivalent replacement
> without material classification loss.

Even then, do not claim:

- complete QKFormer replacement;
- full attention replacement;
- removal of all multiplications;
- measured SRAM, latency, energy, or hardware speedup;
- cross-architecture generalization.

If one category fails twice, retain both rows as a boundary result, record the
root-cause metrics, stop the experiment sequence, and request the user's
scientific judgment.
