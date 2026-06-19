# QK-LUTFormer Experiment Inclusion Audit

Status date: 2026-06-18

The manuscript is organized around one claim chain:

1. QKFormer exposes bounded, repeatable local Q/K address fields.
2. Correct address semantics improve held-out response reconstruction beyond
   coarse and shuffled controls.
3. Support-aware and subspace lookup retain this signal without materializing
   raw full-vector Q/K tables.

Downstream accuracy is a separate gate, not evidence for the first three
claims.

## Main Paper

| Evidence | Paper role | Status |
|---|---|---|
| E0 occupancy and conditional variance | Establish that compact addresses are populated and auditable | Include as one compact table |
| E1 correct/token-channel/shuffled reconstruction | Central address-semantics evidence | Include as the main quantitative result |
| CIFAR-10 checkpoint replication | Show that E1 is not a single-backbone artifact | Include |
| CIFAR-100 calibration-size seeds 42--44 | Support calibration-efficiency for reconstruction | Include as a mean/std curve |
| E5 hierarchical backoff | Make misses and fallback behavior explicit | Include, but label compactness gate as partial |
| E7 subspace-decoupled QK-LUT | Main compact lookup contribution under the 25% entry gate | Include |
| Seed-42 E7 component/footprint ablation | Explain the compact method with one fixed analysis cell | Include as the main method table and Figure 5 |
| Calibrated QKFormer current replacement | Close part of the reconstruction-to-classification fidelity gap | Include as downstream support, not the method verdict |
| All-attention grouped projection LUT | Show simultaneous fidelity across all four attention projection currents | Include as completeness evidence; do not present as standalone novelty |

## Supplementary Material

| Evidence | Reason to retain outside the eight-page main paper |
|---|---|
| Per-stage and per-module E1 tables | Useful reviewer audit trail; too granular for the main claim |
| E2 stage/calibration/blend replacement sweeps | Valid diagnostic showing why reconstruction does not imply end-to-end replacement |
| E5 fallback fractions for every size and seed | Reproducibility and cold-address analysis |
| E6 fixed-budget hierarchy failure | Negative evidence motivating subspace decomposition |
| E4 factorized adapter pilot | Failed alternative; useful for a journal extension or detailed limitations |
| Full machine-readable seed rows and protocol metadata | Reproducibility |
| Selective-oracle per-sample margin bins | Useful diagnostic, but label-informed and not a deployable method | Keep outside the main paper |
| Classification adapter diagnostics | Downstream utility is outside the current method claim | Retain only as project history |
| All-affine scalar LUT audit | Final uniform 0.50 pp criterion preserves Q/K/V and all 32 Conv/Linear outputs | Retain as exploratory fidelity evidence, not a compactness or hardware result |

## Exclude From Scientific Evidence

| Evidence | Exclusion reason |
|---|---|
| Random-initialized or silent E0 smoke runs | Engineering checks only |
| All E3 runs produced before protocol version 2 | `adapter.train()` recursively switched the frozen backbone to train mode and changed BatchNorm running statistics |
| Protocol-v2/v3 CIFAR-100 T=4 exploratory runs | Backbone mode was fixed, but evaluation did not yet match the upstream AMP/timm validation protocol |
| Post-hoc alpha, residual-scale, stage, checkpoint, or seed selection | Would constitute parameter shopping rather than a registered method comparison |
| Selective-oracle accuracy as a method result | It uses validation labels and is only an upper-bound diagnostic |

## Claim Boundary

- **Supported:** compact local Q/K addressability, held-out reconstruction,
  address alignment, calibration efficiency, explicit fallback accounting, and
  subspace entry-budget compression.
- **Secondary:** same-architecture CIFAR-100 T=4 classification stability under
  five registered matched-control gates; classification is not the method
  verdict.
- **Do not claim:** measured SRAM/latency/energy, ImageNet generalization, or a
  production full-LUT QKFormer replacement.
