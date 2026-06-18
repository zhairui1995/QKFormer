# CIFAR-100 All-Attention Projection LUT Replacement

Date: 2026-06-18

## Scope

- Architecture: QKFormer CIFAR-100, seed 42.
- Checkpoints: independently trained `T=1` and `T=4` checkpoints.
- Replaced targets:
  - `stage1.0.tssa`
  - `stage2.0.tssa`
  - `stage3.0.ssa`
  - `stage3.1.ssa`
- Evaluation: complete CIFAR-100 validation split.
- Replacement point: every attention module's `proj_conv` current, before the
  original `proj_bn` and `proj_lif`.

This is all-attention projection replacement. It is not replacement of the
Q/K/V projections, MLPs, patch embeddings, BN/LIF dynamics, classifier, or the
entire QKFormer network.

## Prototype-LUT Boundary

The original scalar response-prototype LUT does not extend to all four targets:

| Method | T=1 Acc@1 | T=1 drop | T=4 Acc@1 | T=4 drop |
|---|---:|---:|---:|---:|
| Clean hook | 77.78 | 0.00 | 81.09 | 0.00 |
| Static aligned prototype | 31.07 | 46.71 | 5.44 | 75.65 |
| Moment/TCSLU prototype | 56.28 | 21.50 | 54.08 | 27.01 |

The failure is concentrated in the SSA projection currents. On CIFAR-100
`T=1`, replacing `stage3.0.ssa` or `stage3.1.ssa` alone with the calibrated
Q/K prototype loses 7.85 and 8.41 percentage points, respectively. Increasing
address-bin resolution or sequentially calibrating downstream layers does not
close the gap.

## Grouped Spike-Pattern Projection LUT

For binary projection input `x`, split the input channels into `g`-bit groups.
For each group, precompute the contribution of every binary pattern to every
output channel. The original 1x1 projection is reconstructed as a sum of table
lookups plus the bias:

`y = bias + sum_group LUT_group[pattern_group(x)]`.

This is an algebraic decomposition of the frozen projection weights rather than
a response-prototype approximation. The experiment still executes the original
`proj_conv` before the hook, so the present result validates numerical and
classification fidelity; it is not a measured runtime speedup.

## Full-Validation Results

| Group bits | T=1 Acc@1 | T=1 delta | T=4 Acc@1 | T=4 delta | FP32 values | FP32 footprint |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 77.73 | -0.05 | 81.15 | +0.06 | 10,911,744 | 42,624 KiB |
| 4 | 77.74 | -0.04 | 81.11 | +0.02 | 1,363,968 | 5,328 KiB |
| 2 | 77.72 | -0.06 | 81.09 | 0.00 | 681,984 | 2,664 KiB |

Positive T=4 deltas are treated as no-loss fluctuations, not accuracy gains.
The 2-bit configuration is the most compact grouped-LUT row tested.

For the 2-bit configuration:

- aggregate current MSE:
  - T=1: `2.0974e-9`
  - T=4: `1.8155e-9`
- aggregate current cosine is effectively `1.0`;
- hit rate is `1.0` because all binary group patterns have deterministic table
  entries;
- FP16 value storage is 1,332 KiB.

## Analytical Operation Proxy

For one time step, the four replaced projections contain 2,359,296 MACs per
sample. The grouped implementation performs:

| Group bits | Vector lookups / sample / step | Lookup-to-MAC count ratio | Scalar table values read / original MAC |
|---:|---:|---:|---:|
| 8 | 1,536 | 0.000651 | 0.125 |
| 4 | 3,072 | 0.001302 | 0.250 |
| 2 | 6,144 | 0.002604 | 0.500 |

Each vector lookup returns all output-channel contributions for one input
group. These ratios are analytical counts only. They exclude address
generation, table bandwidth, accumulation cost, cache behavior, BN/LIF, and
measured hardware cycles or energy.

## Traceable Results

- Reproduction entrypoint:
  `scripts/server/run_qkformer_grouped_projection_all4.sh`
- 8-bit:
  - `results/qk_grouped_projection_c100_t1_all4_full_20260618_185121`
  - `results/qk_grouped_projection_c100_t4_all4_full_20260618_185121`
- 4-bit:
  - `results/qk_grouped_projection_g4_c100_t1_all4_full_20260618_185321`
  - `results/qk_grouped_projection_g4_c100_t4_all4_full_20260618_185321`
- 2-bit:
  - `results/qk_grouped_projection_g2_c100_t1_all4_full_20260618_185455`
  - `results/qk_grouped_projection_g2_c100_t4_all4_full_20260618_185455`

## Verdict

`PASS` for all-attention projection-current LUT replacement on QKFormer
CIFAR-100 `T=1` and `T=4`.

Claim unlocked: every attention projection current in this QKFormer
configuration can be replaced by a grouped binary-pattern LUT with no material
classification loss.

Claim not unlocked: complete-network LUT replacement, cross-architecture
generalization, measured latency, measured SRAM, measured energy, or hardware
speedup.
