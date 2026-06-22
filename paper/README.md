# QK-LUTFormer Paper Draft

This directory contains a LaTeX paper draft for the QK-LUTFormer / QKFormer-LUT
hybrid branch.

Current paper identity:

- scoped `GO-METHOD`: QK-LUTFormer studies when spiking attention is
  lookup-addressable through semantic Q/K alignment, global-residual backoff,
  compact subspace lookup, and TCSLU dynamic calibration.
- Current title: `QK-LUTFormer: Semantic Lookup Addressability and Dynamic
  Calibration for Spiking Attention`.
- `E1 PASS`: correct Q/K addresses beat global, token/channel, and shuffled
  controls across CIFAR-10 checkpoints and CIFAR-100 calibration seeds.
- `E5 PARTIAL`: hierarchical backoff QK-LUT passes reconstruction and fallback
  interpretability gates but fails the compact-table compression gate.
- `E7 PASS`: subspace-decoupled residual QK-LUT beats token/channel and
  shuffled-subspace controls across CIFAR-100 calibration sizes/seeds while
  staying below the 25% entry-budget gate.
- New component and footprint analysis is fixed to CIFAR-100 T=1 seed 42,
  128 calibration batches, and full validation.
- The completed five-gate CIFAR-100 T=4 classification study is secondary,
  protocol-scoped evidence; it is not cross-architecture or cross-dataset
  generalization and is not the basis of the method verdict.
- All four attention `proj_conv` currents have a near-lossless grouped-LUT
  replacement result on CIFAR-100 `T=1/4`. This is completeness/fidelity
  support, not complete-network replacement or measured acceleration.
- The final 0.50-point near-lossless criterion also preserves a cumulative
  substitution of all 32 Conv1d/Conv2d/Linear outputs
  on CIFAR-100 `T=1`.
- The latest native audit removes all 32 Conv/Linear affine operators, all 31
  paired inference BN modules, and all 35 LIF state transitions from the
  evaluated QKFormer runtime path. Full-validation drops are 0.20 pp at `T=1`
  and 0.22 pp at `T=4`. This is retained as a completeness boundary rather
  than a core contribution.
- CIFAR10-DVS deployment is complete below the registered 84.0% gate. Route 1
  is the main event-data result at 83.9% LUT-only Acc@1. The later
  user-authorized SpiLiFormer Route-3 extension reaches 81.7% from an 81.2%
  from-scratch teacher; it is bounded cross-architecture implementation
  evidence and does not reproduce the published 86.7% clean result.

The clean mother draft is organized by research questions rather than the
historical experiment sequence. Detailed restructure markers and evidence
movement are recorded in `PAPER_RESTRUCTURE_REVIEW.md`. The pre-restructure
LaTeX source is preserved under `backups/pre_restructure_20260621/`.

Current claim boundary:

- Supported: address-specific held-out response reconstruction, calibration
  data-efficiency evidence on CIFAR-100, explicit fallback accounting, and
  negative controls showing correct address alignment matters.
- Supported: compact subspace-decoupled reconstruction under a 25% entry-budget
  gate.
- Partially supported: hierarchical backoff as a scalable audit mechanism.
- Secondary: same-architecture CIFAR-100 T=4 classification stability under
  the registered matched-control gates.
- Secondary/completeness: LUT-native replacement fidelity for learned affine,
  paired BN, and LIF-transition operator classes in QKFormer.
- Not supported: energy, latency, measured SRAM, ImageNet, hardware
  acceleration, replacement of pooling/residual/SSA matrix-product reductions,
  or a full production graph-level LUT wrapper.

See `PAPER_STATUS_AND_EXPERIMENT_PLAN.md` for the latest experiment digest and
paper-directed next experiment priorities.

Compile from this directory with:

```bash
latexmk -pdf main.tex
```

or use any equivalent LaTeX engine available locally.

Generate the E7 component figure with:

```bash
python3 ../scripts/plot_qk_lut_e7_component_ablation.py
```
