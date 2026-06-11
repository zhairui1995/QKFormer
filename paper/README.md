# QK-LUTFormer Paper Draft

This directory contains a LaTeX paper draft for the QK-LUTFormer / QKFormer-LUT
hybrid branch.

Current paper-side verdict:

- `GO-AUDIT`: the strongest supported paper is a structured Q/K
  lookup-addressability audit.
- Current title: `QK-LUTFormer: Auditing Hierarchical Lookup Addresses in
  Spiking Q-K Attention`.
- `E1 PASS`: correct Q/K addresses beat global, token/channel, and shuffled
  controls across CIFAR-10 checkpoints and CIFAR-100 calibration seeds.
- `E5 PARTIAL`: hierarchical backoff QK-LUT passes reconstruction and fallback
  interpretability gates but fails the compact-table compression gate.
- `E4 FAIL/PENDING`: factorized LUT modes did not pass the CIFAR-100 pilot and
  have not promoted the paper to a method track.

Current claim boundary:

- Supported: address-specific held-out response reconstruction, calibration
  data-efficiency evidence on CIFAR-100, explicit fallback accounting, and
  negative controls showing correct address alignment matters.
- Partially supported: hierarchical backoff as a scalable audit mechanism and
  low-disturbance residual adapter utility as a stress test.
- Not supported: stable accuracy improvement, compact-table compression,
  energy, latency, ImageNet, measured hardware acceleration, or a full
  production LUT wrapper.

See `PAPER_STATUS_AND_EXPERIMENT_PLAN.md` for the latest experiment digest and
paper-directed next experiment priorities.

Compile from this directory with:

```bash
latexmk -pdf main.tex
```

or use any equivalent LaTeX engine available locally.
