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
- `CIFAR-100 T=4 CONDITIONAL`: the centered aligned residual reaches a best
  run of 81.56% versus the 81.23% saved checkpoint, but averages 81.22% over
  three seeds. Its accuracy-oracle advantage over shuffled lookup is only
  0.04 points, so this is a limits result rather than stable method gain.
- `E4 FAIL/PENDING`: factorized LUT modes did not pass the CIFAR-100 pilot and
  have not promoted the paper to a method track.

Current claim boundary:

- Supported: address-specific held-out response reconstruction, calibration
  data-efficiency evidence on CIFAR-100, explicit fallback accounting, and
  negative controls showing correct address alignment matters.
- Supported: compact subspace-decoupled reconstruction under a 25% entry-budget
  gate.
- Partially supported: hierarchical backoff as a scalable audit mechanism.
- Limits evidence: frozen residual adapters test downstream transfer but do not
  currently establish stable accuracy improvement.
- Not supported: stable accuracy improvement, energy, latency, ImageNet,
  measured hardware acceleration, or a full production LUT wrapper.

See `PAPER_STATUS_AND_EXPERIMENT_PLAN.md` for the latest experiment digest and
paper-directed next experiment priorities.

Compile from this directory with:

```bash
latexmk -pdf main.tex
```

or use any equivalent LaTeX engine available locally.
