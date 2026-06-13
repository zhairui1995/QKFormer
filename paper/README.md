# QK-LUTFormer Paper Draft

This directory contains a LaTeX paper draft for the QK-LUTFormer / QKFormer-LUT
hybrid branch.

Current paper-side verdict:

- `GO-AUDIT`: the strongest supported paper is a structured Q/K
  lookup-addressability audit.
- Current title: `QK-LUTFormer: Compact and Auditable Lookup Addresses for
  Spiking Q-K Attention`.
- `E1 PASS`: correct Q/K addresses beat global, token/channel, and shuffled
  controls across CIFAR-10 checkpoints and CIFAR-100 calibration seeds.
- `E5 PARTIAL`: hierarchical backoff QK-LUT passes reconstruction and fallback
  interpretability gates but fails the compact-table compression gate.
- `E7 PASS`: subspace-decoupled residual QK-LUT beats token/channel and
  shuffled-subspace controls across CIFAR-100 calibration sizes/seeds while
  staying below the 25% entry-budget gate.
- `CIFAR-100 T=4 CHECKPOINT PASS`: a matched one-epoch deterministic gate
  raises aligned mean Acc@1 from 81.22% to 81.73% over three adapter seeds.
  Shuffled and global controls reach 81.62% and 81.46%; independent-backbone
  replication is running before any broader claim.
- `E4 FAIL/PENDING`: factorized LUT modes did not pass the CIFAR-100 pilot and
  have not promoted the paper to a method track.

Current claim boundary:

- Supported: address-specific held-out response reconstruction, calibration
  data-efficiency evidence on CIFAR-100, explicit fallback accounting, and
  negative controls showing correct address alignment matters.
- Supported: compact subspace-decoupled reconstruction under a 25% entry-budget
  gate.
- Partially supported: hierarchical backoff as a scalable audit mechanism.
- Supported on one checkpoint: disjoint-calibration selective utility with
  matched global and shuffled controls.
- Not supported: backbone-stable accuracy improvement, energy, latency, ImageNet,
  measured hardware acceleration, or a full production LUT wrapper.

See `PAPER_STATUS_AND_EXPERIMENT_PLAN.md` for the latest experiment digest and
paper-directed next experiment priorities.

Compile from this directory with:

```bash
latexmk -pdf main.tex
```

or use any equivalent LaTeX engine available locally.
