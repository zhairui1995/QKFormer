# QK-LUTFormer Paper Status And Experiment Plan

Status date: 2026-06-11

## Current Editorial Decision

**Decision: `GO-AUDIT`.**

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
| 8 | 3.8001% | 3.3315% | 99.7952% |
| 32 | 4.2756% | 3.4084% | 99.9557% |
| 128 | 4.4243% | 3.4253% | 99.9908% |
| 512 | 4.4674% | 3.4274% | 99.9981% |

Paper meaning:

- `Supported`: the address reconstruction effect is not fragile to large
  calibration budgets.
- `Supported`: eight batches recover about 85% of the 512-batch address gain.
- `Do Not Claim`: this is not a measured latency, memory, energy, or hardware
  efficiency result.

### E3 / Downstream Utility

E3 remains a limits result.

- Address LUT can beat token/channel and shuffled controls in some settings.
- Global smoothing is often stronger for classification and loss.
- Aligned-vs-shuffled ordering is not scale-stable.

Paper meaning:

- `Partially Supported`: address structure can be used in low-disturbance
  adapter probes.
- `Do Not Claim`: stable top-1 improvement or method superiority.

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

## Current Manuscript Implications

The current `main.tex` title is:

**QK-LUTFormer: Auditing Hierarchical Lookup Addresses in Spiking Q-K Attention**

This is acceptable for the audit track: it keeps the QK-LUTFormer brand while
avoiding a production-wrapper or stable-accuracy promise. The abstract and
experiment section include E1, E5, and the downstream utility limits.

## Next Experiments For The Server Agent

These are paper-directed top-level experiments. The server agent should decide
implementation details and return machine-readable summaries.

### Priority 1: Replicate Calibration-Size Sweep Across C100 Seeds

Goal: make the calibration-efficiency claim reviewer-proof.

Current status: calibration-size sweep is only on CIFAR-100 calibration seed 42.

Run:

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

Paper use:

- If pass: keep calibration-size table as a main or appendix robustness result.
- If fail: report seed-42 calibration-size sweep as secondary only; do not claim
  calibration efficiency.

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

### Priority 3: Budgeted / Compressed Hierarchy

Goal: test whether support-threshold budgeting can satisfy the E5 compression
gate without losing address reconstruction.

Run:

- CIFAR-100 T=1.
- Calibration sizes: 8, 32, 128, 512.
- Calibration seeds: 42, 43, 44.
- Support thresholds: min-count sweep such as 4, 8, 16, 32, 64.

Gate:

- A budgeted setting should exist for every size/seed with effective entries
  <= 25% of compact full-address space.
- It should retain at least 80% of full-address reconstruction gain.
- It must beat token/channel and remain above shuffled controls.

Paper use:

- If pass: promote compressed hierarchy to the main method claim.
- If partial/fail: keep E5 as an audit result and move compression to appendix
  or future work.

### Priority 4: CIFAR-100 T=4 Accuracy-Oriented Validation

Goal: obtain a fair T=4 CIFAR-100 baseline and test whether a pre-registered
LUT variant can improve it while preserving interpretability.

Run:

- Train QKFormer CIFAR-100 T=4 from scratch.
- Then run E0/E1 and a small, pre-registered LUT adapter/control sweep on the
  best checkpoint.
- Keep global, token/channel, and shuffled controls adjacent to any reported
  address result.

Paper use:

- If address LUT beats the T=4 baseline and controls under the gate, report as
  downstream utility evidence.
- If not, keep as appendix/limits evidence and do not weaken the reconstruction
  claim with seed shopping.

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

Not allowed:

- Stable accuracy improvement.
- Compact-table compression before E6 passes.
- ImageNet generalization.
- Energy, latency, or hardware acceleration.
- Full LUT replacement of QKFormer.
- Method superiority of factorized LUT before E4 gate passes.
