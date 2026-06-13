# QK-LUTFormer Experiment Inclusion Audit

Status date: 2026-06-12

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
| CIFAR-100 T=4 deterministic margin gate | Checkpoint-level selective utility with disjoint calibration and matched controls | Include as one concise table/paragraph |

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
| Two-epoch deterministic gate | Best run reaches 82.02%, but aligned--global separation misses the predeclared gate | Keep as a limits/robustness row |

## Exclude From Scientific Evidence

| Evidence | Exclusion reason |
|---|---|
| Random-initialized or silent E0 smoke runs | Engineering checks only |
| All E3 runs produced before protocol version 2 | `adapter.train()` recursively switched the frozen backbone to train mode and changed BatchNorm running statistics |
| Protocol-v2/v3 CIFAR-100 T=4 exploratory runs | Backbone mode was fixed, but evaluation did not yet match the upstream AMP/timm validation protocol |
| Post-hoc alpha, residual-scale, stage, or seed selection | Would constitute parameter shopping rather than a registered method comparison |
| Selective-oracle accuracy as a method result | It uses validation labels and is only an upper-bound diagnostic |

## Claim Boundary

- **Supported:** compact local Q/K addressability, held-out reconstruction,
  address alignment, calibration efficiency, explicit fallback accounting, and
  subspace entry-budget compression.
- **Supported on one checkpoint:** the matched one-epoch calibration-only gate
  improves all aligned adapter seeds, reaches 81.7267% mean Acc@1, and exceeds
  global and shuffled controls in the mean.
- **Conditional:** aligned does not beat shuffled for every adapter seed;
  independent CIFAR-100 T=4 backbone seeds 43/44 are still running.
- **Do not claim:** backbone-stable accuracy improvement, measured SRAM/latency/energy,
  ImageNet generalization, or a production full-LUT QKFormer replacement.
