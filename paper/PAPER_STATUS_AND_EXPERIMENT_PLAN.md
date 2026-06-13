# QK-LUTFormer Paper Status And Experiment Plan

Status date: 2026-06-12

## Current Editorial Decision

**Decision: `GO-AUDIT` with checkpoint-conditional downstream utility.**

The paper should currently be written as a structured Q/K lookup-addressability
audit, not as a residual adapter method paper. The central claim is:

> Binary Q/K states in QKFormer form reproducible lookup addresses for local
> response reconstruction, but this structure does not yet imply stable
> downstream classification utility.

This decision incorporates the previous Codex round:

- Created `docs/QK_LUTFORMER_SECOND_PAPER_DECISION_PACKAGE.md`.
- Created `docs/QK_LUTFORMER_REVISION_BLUEPRINT.md`.
- Created a dedicated NotebookLM notebook and ran R1-R4 review prompts.
- Implemented E4 modes and gates: `factorized_sum_lut`,
  `factorized_gated_lut`, `matched_param_address_lut`, and
  `factorized_shuffled_lut`.
- Generated machine reports:
  - `docs/QK_LUTFORMER_E1_GATE_REPORT.md`: `PASS`.
  - `docs/QK_LUTFORMER_E4_GATE_REPORT.md`: `FAIL/PENDING`.
  - `results/qk_lutformer_e5_hierarchical_backoff_report.md`: `PARTIAL`.
  - `results/qk_lutformer_e7_subspace_report.md`: `PASS`.

## Latest Experiment Digest

### E1 Gate

E1 is now the main paper evidence.

| Setting | Correct address | Token/channel | Shuffled address | Interpretation |
|---|---:|---:|---:|---|
| CIFAR-10 T=1 checkpoint 42 | 3.7597% | 2.8026% | -18.8859% | Address beats coarse control. |
| CIFAR-10 T=1 checkpoint 43 | 5.9361% | 4.4966% | -17.5642% | Strongest C10 checkpoint. |
| CIFAR-10 T=1 checkpoint 44 | 4.6503% | 3.6514% | -17.0855% | Replicates address > token/channel. |
| CIFAR-100 T=1 calib seed 42 | 4.4243% | 3.4253% | -15.7967% | Stable C100 result. |
| CIFAR-100 T=1 calib seed 43 | 4.4207% | 3.4225% | -16.5924% | Stable C100 result. |
| CIFAR-100 T=1 calib seed 44 | 4.4292% | 3.4261% | -15.0756% | Stable C100 result. |

Paper meaning:

- `Supported`: address-specific reconstruction is real and reproducible.
- `Supported`: Q/K detail adds about 1.0 to 1.13 percentage points of relative
  MSE reduction beyond token/channel coarsening.
- `Supported`: shuffled controls strongly break reconstruction, so table
  capacity alone is not enough.

### CIFAR-100 Calibration-Size Sweep

| Calibration batches | Correct address | Token/channel | Validation hit rate |
|---:|---:|---:|---:|
| 8 | 3.7840% | 3.3258% | 99.7982% |
| 32 | 4.2671% | 3.4027% | 99.9558% |
| 128 | 4.4247% | 3.4246% | 99.9907% |
| 512 | 4.4752% | 3.4386% | 99.9981% |

Values are means over calibration seeds 42--44; the full validation split is
used for every run.

Paper meaning:

- `Supported`: the address reconstruction effect is not fragile to large
  calibration budgets.
- `Supported`: eight batches recover about 85% of the 512-batch address gain.
- `Do Not Claim`: this is not a measured latency, memory, energy, or hardware
  efficiency result.

### E3 / Downstream Utility

All E3 accuracy runs produced before protocol version 2 are excluded from
scientific evidence. The adapter module had registered the frozen backbone as
a child module, so calling `adapter.train()` also switched backbone BatchNorm
layers to training mode and changed their running statistics. Protocol versions
2--3 fix the backbone mode but do not reproduce the upstream AMP/timm validation
loader, so they remain engineering diagnostics only.

The initial admissible downstream evidence is the completed CIFAR-100 $T=4$
protocol-v4 two-epoch gate: frozen backbone in evaluation mode, an $\alpha=0$ identity
preflight, upstream AMP/timm full validation, fixed stage1, fixed
$\alpha=0.025$, two epochs, 128 calibration/training batches, and seeds
42--44.

- Paired QKFormer reevaluation: 81.21% Acc@1; saved checkpoint best: 81.23%.
- Global + centered address: 81.35 / 81.56 / 80.75, mean 81.22%.
- Matched centered shuffled: 81.25 / 80.95 / 80.87, mean 81.02%.
- Global mean: 81.16 / 81.21 / 81.08, mean 81.15%.
- Accuracy-oracle gains: aligned 3.37, shuffled 3.33, global 3.21 points.

The subsequent matched one-epoch deterministic gate uses disjoint prototype,
adapter, and gate-calibration train partitions. Its inference score is the
LUT-minus-baseline top1/top2 margin difference; validation labels are not used
to fit the threshold.

- Paired baseline: 81.22%.
- Aligned gated seeds 42/43/44: 81.69 / 81.93 / 81.56, mean 81.7267%.
- Shuffled gated: 81.88 / 81.43 / 81.54, mean 81.6167%.
- Global gated: 81.61 / 81.27 / 81.51, mean 81.4633%.
- Aligned minimum delta over baseline: +0.34 points.
- Two-epoch aligned gated best: 82.02%; mean 81.7467%, but the aligned--global
  mean gap is only 0.0434 points.

Paper meaning: the one-epoch gate passes the checkpoint-level acceptance rule
and supports calibrated selective utility. It does not yet support a
backbone-stable address-specific accuracy claim because aligned loses to
shuffled for one adapter seed and independent backbone seeds are pending.

### E4 / Factorized Method Track

E4 is a negative/insufficient method-track result.

- Implementation and gate are ready.
- Current gate decision is not `GO-METHOD`.
- Factorized modes should stay in the appendix or limits discussion unless a
  later pre-registered gate passes.

### E5 / Hierarchical Backoff

E5 answers the reviewer concern that direct full Q/K vector lookup can become a
dimensionally explosive table.

| Calibration batches | Hierarchical | Full address | Token/channel | Effective entries |
|---:|---:|---:|---:|---:|
| 1 | 0.3178% | 0.4179% | 2.5493% | 0.3305 |
| 2 | 2.1445% | 2.1789% | 3.0002% | 0.3566 |
| 4 | 3.1962% | 3.2150% | 3.2261% | 0.3808 |
| 8 | 3.7742% | 3.7840% | 3.3258% | 0.4040 |
| 32 | 4.2667% | 4.2671% | 3.4027% | 0.4477 |
| 128 | 4.4248% | 4.4247% | 3.4246% | 0.4822 |
| 512 | 4.4752% | 4.4752% | 3.4386% | 0.5107 |

Paper meaning:

- `Supported`: fallback behavior is explicit and measurable; fallback
  fractions sum to one for every run.
- `Supported`: from 8 batches onward, hierarchical backoff beats token/channel
  for every seed and preserves the full-address reconstruction gain.
- `Supported`: shuffled full-address lookup remains much worse, so alignment
  still matters.
- `Partially Supported`: hierarchical backoff is a scalable audit mechanism.
- `Do Not Claim`: it does not yet solve compact-table compression, because
  effective supported entries remain above the 25% gate.

### E7 / Subspace-Decoupled Compact LUT

E7 is the current compact-table positive result. It keeps the reconstruction
claim but replaces monolithic full-address storage with additive sub-address
residual subtables.

| Calibration batches | Token/channel | TC+Q/G | TC+Q/K | TC+Q/G entries | TC+Q/K entries |
|---:|---:|---:|---:|---:|---:|
| 8 | 3.3258% | 3.7502% | 4.0088% | 0.0883 | 0.2020 |
| 32 | 3.4027% | 3.8463% | 4.1357% | 0.0883 | 0.2047 |
| 128 | 3.4246% | 3.8744% | 4.1712% | 0.0883 | 0.2054 |
| 512 | 3.4386% | 3.8884% | 4.1866% | 0.0883 | 0.2057 |

Paper meaning:

- `Supported`: token/channel+Q/gate subspace residual LUT beats token/channel
  for every CIFAR-100 calibration size and seed while using only 8.83% of
  supported full-address entries.
- `Supported`: token/channel+Q/K subspace LUT remains below the 25% entry
  budget and retains most full-address reconstruction gain.
- `Supported`: shuffled-subspace controls fail in every run, so the result is
  not just residual table capacity.
- `Do Not Claim`: this is an entry-budget and reconstruction result, not a
  measured SRAM, latency, energy, or classification-improvement result.

## Current Manuscript Implications

The current `main.tex` title is:

**QK-LUTFormer: Compact and Auditable Lookup Addresses for Spiking Q-K Attention**

This is acceptable for the audit track: it keeps the QK-LUTFormer brand while
avoiding a production-wrapper or stable-accuracy promise. The abstract and
experiment section now include E1, E5, E7, and the downstream utility limits.

## Next Experiments For The Server Agent

These are paper-directed top-level experiments. The server agent should decide
implementation details and return machine-readable summaries.

### Priority 1: Replicate Calibration-Size Sweep Across C100 Seeds

Goal: make the calibration-efficiency claim reviewer-proof.

Current status: **completed and passed** across calibration seeds 42--44.

Completed protocol:

- CIFAR-100 T=1.
- Calibration sizes: 8, 32, 512.
- Calibration seeds: 43 and 44.
- Full validation.
- Same controls: global, candidate/background, token/channel, correct address,
  shuffled address.

Acceptance:

- Correct address beats token/channel at every size and seed.
- Shuffled address remains worse than global.
- Eight-batch setting remains meaningfully positive.

Paper use: retain the mean/std curve in the main paper as calibration-efficiency
evidence for reconstruction only.

### Priority 2: Component Ablation For Address Semantics

Goal: show what part of the address carries the extra signal beyond
token/channel.

Run E1-style reconstruction controls that progressively add address components:

- head + token + channel only.
- plus Q bit / gate bit.
- plus K bit.
- full Q/K/population address.
- shuffled full address.

Acceptance:

- Full address should beat every strict subset.
- At least one Q/K-specific increment should be positive beyond
  token/channel.

Paper use:

- This directly strengthens novelty against “this is just token/channel
  binning” and “ordinary lookup table capacity” objections.

### Priority 3: Fixed-Budget / Compressed Hierarchy

Status: superseded as the main compression story by E7. E6 fixed-budget
hierarchical pruning failed the compact gate, while E7 subspace decomposition
passed it. Keep E6 as appendix/negative evidence if space permits.

Historical goal: test whether a fixed-entry-budget hierarchy can satisfy the E5
compression gate without losing address reconstruction. This was the direct
reply to the reviewer concern that full Q/K lookup tables can explode in size.

Current implementation:

- `tools/qkformer_lut_e1_recon.py` supports
  `QKFORMER_LUT_E1_HIERARCHY_BUDGET_FRACTION`.
- `scripts/server/run_qkformer_lut_e6_budget_fraction_sweep.sh` runs the
  fixed-budget sweep.
- Default policy is `balanced_quota`: keep a small quota for token/channel
  fallback and reserve the rest for progressively more Q/K-specific levels.

Run:

- CIFAR-100 T=1.
- Calibration sizes: 8, 32, 128, 512.
- Calibration seeds: 42, 43, 44.
- Budget fraction: 0.25 of the compact full-address space.
- Support thresholds: min-count sweep such as 2, 4, 8 under the same fixed
  budget.

Gate:

- A fixed-budget setting should exist for every size/seed with effective
  entries <= 25% of compact full-address space.
- It should retain at least 80% of full-address reconstruction gain.
- It must beat token/channel and remain above shuffled controls.

Paper use:

- Keep E6 as a limits result: pruning the full-address hierarchy is not enough.
- Promote E7 subspace decomposition as the compact reconstruction result.

### Priority 4: CIFAR-100 T=4 Accuracy-Oriented Validation

Status: **completed; conditional/limits result.**

Goal: obtain a fair T=4 CIFAR-100 baseline and test whether a pre-registered
LUT variant can improve it while preserving interpretability.

Run:

- Train QKFormer CIFAR-100 T=4 from scratch.
- Then run E0/E1 and a small, pre-registered LUT adapter/control sweep on the
  best checkpoint.
- Keep global, token/channel, and shuffled controls adjacent to any reported
  address result.

Paper use: retain one concise main-paper table because the gate is the honest
downstream test of the reconstruction claim. Put per-sample oracle bins and old
exploratory sweeps in the appendix. Do not weaken the reconstruction claim with
seed shopping.

### Priority 5: Do Not Run More Unregistered Accuracy Tuning

Stop alpha, residual-scale, seed shopping, and stage shopping unless tied to a
pre-registered diagnostic question.

Reason:

- The paper already has enough evidence that classification utility is unstable.
- More unconstrained tuning will weaken the claim boundary and invite reviewer
  criticism.

## Recommended Paper Work Before New Results Return

1. Expand Related Work into four groups: SNN attention, LUT-native networks,
   discrete codebooks/retrieval, and adapters.
2. Add a Discussion subsection: why local reconstruction does not guarantee
   downstream classification utility.
3. Add appendix tables for E3/E4 limits and E5 per-seed budget details.
4. Keep all hardware metrics as proxies only.

## Final Claim Boundary

Allowed:

- Q/K-derived binary addresses are structured and predictive of local responses.
- Correct address alignment matters beyond token/channel coarsening.
- The reconstruction signal is stable across C10 checkpoints and C100
  calibration seeds.
- Low calibration budgets can preserve much of the C100 reconstruction effect,
- and E5 shows support-aware fallback can make misses explicit.
- E7 supports compact subspace-decoupled reconstruction under an entry-count
  budget.

Not allowed:

- Stable accuracy improvement.
- Measured memory, SRAM, energy, or latency compression from the entry-budget
  proxy alone.
- ImageNet generalization.
- Hardware acceleration.
- Full LUT replacement of QKFormer.
- Method superiority of factorized LUT before E4 gate passes.
