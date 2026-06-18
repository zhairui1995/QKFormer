# QK-LUTFormer Paper Draft

This directory contains a LaTeX paper draft for the QK-LUTFormer / QKFormer-LUT
hybrid branch.

Current paper-side verdict:

- scoped `GO-METHOD`: the supported method is address-specific reconstruction,
  explicit hierarchical backoff, and compact subspace-decoupled lookup.
- Current title: `QK-LUTFormer: Compact and Auditable Lookup Addresses for
  Spiking Q-K Attention`.
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

Current claim boundary:

- Supported: address-specific held-out response reconstruction, calibration
  data-efficiency evidence on CIFAR-100, explicit fallback accounting, and
  negative controls showing correct address alignment matters.
- Supported: compact subspace-decoupled reconstruction under a 25% entry-budget
  gate.
- Partially supported: hierarchical backoff as a scalable audit mechanism.
- Secondary: same-architecture CIFAR-100 T=4 classification stability under
  the registered matched-control gates.
- Not supported: energy, latency, measured SRAM, ImageNet, hardware
  acceleration, or a full production LUT wrapper.

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
