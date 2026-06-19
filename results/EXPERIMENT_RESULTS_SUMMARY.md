# QK-LUTFormer Experiment Results Summary

Updated: 2026-06-18

This is the default low-token result entry point. Do not open raw result JSON
unless a numerical claim needs forensic verification.

## Current Verdict

`GO-METHOD`.

Structured Q/K addressability, held-out response reconstruction, transparent
backoff, compact subspace-decoupled lookup, and same-architecture
multi-checkpoint classification stability are all supported.

## Baselines

| Dataset / setting | Best Acc@1 |
|---|---:|
| CIFAR-10, T=4, seed 42 | 96.08% |
| CIFAR-10, T=1, seed 42 | 95.20% |
| CIFAR-10, T=1, seed 43 | 95.04% |
| CIFAR-10, T=1, seed 44 | 95.08% |
| CIFAR-100, T=1, seed 42 | 77.76% |
| CIFAR-100, T=4, seed 42 | 81.23% |
| CIFAR-100, T=4, seed 43 | 81.35% |
| CIFAR-100, T=4, seed 44 | 81.05% |

## Main Positive Evidence

### E1 Address Reconstruction

- CIFAR-100 T=1, 128 calibration batches: address LUT mean relative MSE
  reduction 4.4247%, token/channel 3.4246%, shuffled address -15.8216%.
- CIFAR-10 T=4: address 3.7080%, token/channel 2.7941%, shuffled -17.8260%.
- CIFAR-10 T=1 checkpoints 42/43/44: address beats token/channel on every
  checkpoint; mean extra reduction is 1.1318 percentage points.
- CIFAR-100 calibration sizes 8/32/128/512 and seeds 42/43/44 preserve the
  address-over-token/channel ordering.

Conclusion: correct Q/K address semantics robustly improve held-out response
reconstruction. This claim is supported.

### E5 Hierarchical Backoff

- From eight calibration batches onward, support-aware hierarchy beats
  token/channel for every seed.
- Misses and fallback levels are explicit.
- Effective entries remain roughly 40.4% to 51.1% of supported full-address
  entries, missing the 25% compactness target.

Conclusion: transparent miss handling is supported; strong compactness is only
partially supported.

### E7 Subspace-Decoupled LUT

- `TC+Q/G` beats token/channel and shuffled controls for every calibration
  size/seed using 8.83% of supported full-address entries.
- `TC+QK` preserves the same ordering using about 20.2% to 20.6% entries.
- At 8/32/128/512 calibration batches, `TC+QK` relative MSE reductions are
  4.0088% / 4.1357% / 4.1712% / 4.1866%.

Conclusion: E7 is the strongest compact-LUT result and the main answer to the
dimensional-explosion criticism.

### Canonical Seed-42 E7 Paper Analysis

CIFAR-100 T=1, seed 42, 128 calibration batches, full validation:

| Estimator | MSE reduction | Supported scalar entries | Entry fraction | Gain retention | Idealized FP32 values |
|---|---:|---:|---:|---:|---:|
| token/channel | 3.4253% | 2,052 | 2.95% | 77.42% | 8.0 KiB |
| TC+Q/gate | 3.8749% | 6,148 | 8.83% | 87.58% | 24.0 KiB |
| TC+QK | 4.1727% | 14,303 | 20.54% | 94.31% | 55.9 KiB |
| full address | 4.4243% | 19,258 | 27.66% | 100.00% | 75.2 KiB |

The shuffled TC+Q/gate and TC+QK controls are -1.8908% and -1.6600%,
respectively. Entry fractions use the compact full-address address-space count
of 69,632 as the denominator. FP32 footprints count prototype values and four
global fallback scalars only; they exclude keys, indices, counters, padding,
allocator overhead, and hardware metadata.

## Historical Classification And Gate Evidence

These experiments remain recorded for reproducibility. They are not the core
method claim and do not by themselves establish cross-architecture stability.

### CIFAR-100 T=4 Seed 42, One Epoch

| Mode | Mean Acc@1 | Delta |
|---|---:|---:|
| aligned Q/K LUT | 81.7267% | +0.5067 |
| shuffled LUT | 81.6167% | +0.3967 |
| global mean | 81.4633% | +0.2433 |
| paired baseline | 81.2200% | - |

Ordering: `aligned > shuffled > global > baseline`. Gate: `PASS` for this
checkpoint.

### CIFAR-100 T=4 Seed 42, Two Epochs

| Mode | Mean Acc@1 | Delta |
|---|---:|---:|
| aligned Q/K LUT | 81.7467% | +0.5267 |
| shuffled LUT | 81.2800% | +0.0600 |
| global mean | 81.2333% | +0.0133 |
| paired baseline | 81.2200% | - |

Ordering: `aligned > shuffled > global > baseline`. Gate: `PASS`.

### CIFAR-100 T=4 Seed 44, Two Epochs

| Mode | Mean Acc@1 | Delta |
|---|---:|---:|
| aligned Q/K LUT | 81.6100% | +0.5700 |
| shuffled LUT | 81.3900% | +0.3500 |
| global mean | 81.3433% | +0.3033 |
| paired baseline | 81.0400% | - |

Ordering: `aligned > shuffled > global > baseline`. Gate: `PASS` in the
historical report.

### Seed 43, One Epoch

| Mode | Mean Acc@1 | Delta |
|---|---:|---:|
| aligned Q/K LUT | 81.5700% | +0.2000 |
| shuffled LUT | 81.3733% | +0.0033 |
| baseline | 81.3700% | — |
| global gated | 81.3567% | -0.0133 |

Aligned and shuffled both exceed baseline; global gated falls minimally below
baseline. Gate: `PASS`.

### Seed 44, One Epoch

| Mode | Mean Acc@1 | Delta |
|---|---:|---:|
| aligned Q/K LUT | 81.4067% | +0.3667 |
| global mean | 81.2900% | +0.2500 |
| shuffled LUT | 81.2400% | +0.2000 |
| baseline | 81.0400% | — |

Ordering: `aligned > global > shuffled > baseline`. Gate: `PASS`.

### Cross-Seed Summary

| Seed | Epochs | aligned | shuffled | global | baseline | Delta | Gate |
|------|--------|--------:|---------:|-------:|---------:|------:|------|
| 42 | 1 | 81.73% | 81.62% | 81.46% | 81.22% | +0.51 | PASS |
| 42 | 2 | 81.75% | 81.28% | 81.23% | 81.22% | +0.53 | PASS |
| 43 | 1 | 81.57% | 81.37% | 81.36% | 81.37% | +0.20 | PASS |
| 44 | 1 | 81.41% | 81.24% | 81.29% | 81.04% | +0.37 | PASS |
| 44 | 2 | 81.61% | 81.39% | 81.34% | 81.04% | +0.57 | PASS |

All five classification gates pass. Aligned Q/K LUT consistently achieves the
highest accuracy across seeds and epoch budgets.

## Experiment Policy

- Existing seed-43/44 checkpoints, metrics, reports, and gate artifacts are
  valid historical records and may be read, downloaded, audited, summarized,
  or archived when needed.
- Do not launch new seed-43/44 experiments by default.
- All new paper-mainline analysis uses seed 42 unless a concrete reviewer
  requirement justifies additional replication.
- Do not continue alpha, stage, checkpoint, or random-seed searches.

## Claim Boundary

Supported:

- structured and occupied Q/K lookup addresses;
- address-specific held-out response reconstruction;
- calibration efficiency;
- explicit hierarchical fallback behavior;
- compact subspace-decoupled reconstruction under an entry-count proxy;
- same-architecture CIFAR-100 T=4 classification stability across independent
  QKFormer training seeds/checkpoints;
- exploratory scalar-level substitution fidelity for all 32 Conv1d/Conv2d/
  Linear outputs in one CIFAR-100 `T=1` checkpoint under the final 0.50-pp
  near-lossless criterion.

Not supported:

- measured SRAM, latency, or energy gains;
- ImageNet gain;
- production LUT hardware acceleration;
- a complete LUT replacement for QKFormer;
- broad Spiking Transformer replacement. A separate Spikformer T=4 diagnostic
  is now retained only as a boundary result: learned temporal+channel gating
  reaches 76.28% Acc@1 from an 88.16% clean baseline.

## Recent Replacement-Fidelity Diagnostics

These diagnostics are not part of the core `GO-METHOD` evidence. They inform
the next paper scope decision.

| Architecture | Dataset | T | Best retained diagnostic | Clean Acc@1 | Injected Acc@1 | Interpretation |
|---|---|---:|---|---:|---:|---|
| QKFormer | CIFAR-10 | 1 | moment/TCSLU current LUT on `stage1.0.tssa` | 94.64% | 94.57% | full-validation near-lossless current replacement |
| QKFormer | CIFAR-10 | 4 | moment/TCSLU current LUT on `stage1.0.tssa` | 95.70% | 95.81% | full-validation current replacement preserved; do not claim accuracy gain |
| Spikformer-4-384w | CIFAR-10 | 4 | learned temporal+channel gate | 88.16% | 76.28% | T>1 temporal/channel mismatch boundary |

The QKFormer rows are seed-42, single-layer current-replacement diagnostics;
they support the local Q/K spike-address lookup contract but do not establish a
complete QKFormer replacement or a hardware speedup. Treat the CIFAR-10 T=4
positive delta as no-loss fluctuation, not as an accuracy-gain claim. Detailed
Spikformer diagnostics were compressed into
`results/spikformer_experiment_archive_20260617.zip`; active Markdown/CSV
state retains only the boundary row above. See
`docs/QKFORMER_SPIKFORMER_DIFFERENCE_CONCLUSION.md`.

## TCSLU AAAI2027 Track

The next experiment package is preregistered in
`docs/TCSLU_AAAI2027_EXPERIMENT_PLAN.md`. It reframes the QKFormer/Spikformer
gap as a temporal-channel dynamic-contract question:

- static semantic LUT is promising for QKFormer T=1 current replacement;
- Spikformer T=4 remains a boundary case;
- TCSLU will test semantic LUT + moment matching + temporal gate + channel
  adapter + support-aware backoff under fixed matched controls.

Machine-readable planning/status tables live under `paper/tables/data/` with
the `tcslu_` prefix. Planned rows are not results.

### Completed TCSLU Current-Replacement Main Rows

Full-validation current-replacement rows were run on the 4090 server with
explicit checkpoints and downloaded locally:

| Architecture | Dataset | T | Static hard | Moment/TCSLU hard | Interpretation |
|---|---|---:|---:|---:|---|
| QKFormer | CIFAR-10 | 1 | 94.64 -> 93.64 | 94.64 -> 94.57 | full-validation near-lossless current replacement |
| QKFormer | CIFAR-10 | 4 | 95.70 -> 95.08 | 95.70 -> 95.81 | calibration preserves current replacement; positive delta is noise-level |
| QKFormer | CIFAR-100 | 1 | 77.78 -> 75.55 | 77.78 -> 77.26 | TCSLU keeps drop to 0.52 pp |
| QKFormer | CIFAR-100 | 4 | 81.09 -> 68.02 | 81.09 -> 80.65 | T>1 static LUT collapses; calibrated TCSLU recovers to 0.44 pp drop |
| Spikformer-4-384w | CIFAR-10 | 4 | 88.16 -> 66.47 | 88.16 -> 76.35 | temporal+channel gate halves the gap, but remains far from lossless |

Traceable result directories:

- `results/tcslu_qkf_c100_t1_full_20260618_104451`
- `results/tcslu_qkf_c100_t4_full_20260618_104512`
- `results/tcslu_qkf_c10_t1_full_20260618_161423`
- `results/tcslu_qkf_c10_t4_full_20260618_161423_t4rerun`
- `results/tcslu_spik_c10_t4_boundary_rerun_20260618_105951`

The QKFormer rows support the paper claim that current-level LUT replacement is
not solved by a static table alone, especially at T=4; moment/channel
calibration and support-aware lookup recover the dynamic contract. The
Spikformer row remains a boundary result: learned temporal+channel calibration
improves over static and moment-only LUT, but does not establish broad
cross-architecture lossless replacement.

### Completed CIFAR-10 Shuffled-Address Current-Replacement Control

The CIFAR-10 aligned-vs-shuffled current replacement control was run under the
same seed-42, `stage1.0.tssa`, 128-calibration-batch, full-validation protocol:

| Setting | Static aligned | Static shuffled | TCSLU/moment aligned | TCSLU/moment shuffled | Interpretation |
|---|---:|---:|---:|---:|---|
| QKFormer CIFAR-10 T=1 | 93.64 | 91.48 | 94.57 | 94.37 | aligned improves static by 2.16 pp and calibrated by 0.20 pp |
| QKFormer CIFAR-10 T=4 | 95.08 | 90.57 | 95.81 | 95.31 | aligned improves static by 4.51 pp and calibrated by 0.50 pp |

Traceable result directories:

- `results/tcslu_qkf_c10_t1_shuffled_20260618_163112`
- `results/tcslu_qkf_c10_t4_shuffled_20260618_163112`
- summary memo: `results/tcslu_qkf_c10_shuffled_control_20260618.md`

Interpretation: preserving Q/K spike-address alignment improves end-to-end
current replacement fidelity beyond distribution-matched prototype smoothing.
This supports the semantic-address claim while remaining scoped to a
single-layer current replacement diagnostic.

### Completed CIFAR-10 T=4 Semantic Address Controls

The remaining Table C controls were run for QKFormer CIFAR-10 T=4 under the
same seed-42, `stage1.0.tssa`, 128-calibration-batch, full-validation protocol:

| Control | Static hard Top-1 | TCSLU/moment Top-1 | Interpretation |
|---|---:|---:|---|
| aligned Q/K address | 95.08 | 95.81 | best calibrated fidelity |
| global mean | 91.41 | 95.63 | calibrated but below aligned |
| same-size random table | 91.87 | 95.52 | capacity alone is insufficient |
| token/channel mean | 94.86 | 95.45 | coarse state below aligned |
| shuffled address | 90.57 | 95.31 | broken alignment below aligned |

Traceable result directories:

- `results/tcslu_qkf_c10_t4_token_channel_mean_20260618_164404`
- `results/tcslu_qkf_c10_t4_global_mean_20260618_164404`
- `results/tcslu_qkf_c10_t4_random_table_20260618_165105`
- summary memo: `results/tcslu_qkf_c10_t4_semantic_controls_20260618.md`

Interpretation: Q/K spike-address alignment is the strongest control. The
current-unit replacement result is not explained by generic smoothing, coarse
token/channel means, random same-size capacity, or prototype-distribution
preservation alone.

### CIFAR-10 T=4 Control Replication Across Calibration Subsets

The aligned/global/token-channel comparison was repeated with five fixed
128-batch calibration subsets while keeping the seed-42 checkpoint, target,
full validation set, and all hyperparameters unchanged:

| Method | Mean Acc@1 | Calibration-subset SD |
|---|---:|---:|
| aligned Q/K address | 95.820% | 0.123 pp |
| global mean | 95.630% | 0.000 pp |
| token/channel mean | 95.542% | 0.107 pp |

Paired differences:

- aligned minus global: +0.190 pp, 95% CI [+0.037, +0.343];
- aligned minus token/channel: +0.278 pp, 95% CI [+0.014, +0.542];
- global minus token/channel: +0.088 pp, 95% CI [-0.045, +0.221].

Conclusion: the original 0.18 pp global-over-token result is not a stable
ordering and must not be used as evidence that global smoothing is superior.
Aligned lookup ranks first in all five calibration subsets. Its paired
advantages over global and token/channel are +0.190 pp and +0.278 pp, with
95% CIs [+0.037, +0.343] and [+0.014, +0.542], respectively; both exclude
zero. This is statistically consistent calibration-subset evidence for the
aligned control, but it is not independent-checkpoint evidence. Traceable summary:
`results/tcslu_qkf_c10_t4_control_replication_20260618_181345/replication_summary.md`.

### Completed CIFAR-10 T=4 Follow-Up Batch

Four follow-up groups were completed under the QKFormer CIFAR-10 T=4 seed-42
full-validation protocol. Summary memo:
`results/tcslu_qkf_c10_t4_followup_experiments_20260618.md`.

| Group | Best / key result | Verdict |
|---|---|---|
| Semantic controls | aligned TCSLU 95.81 beats global 95.63, random 95.52, token/channel 95.45, shuffled 95.31 | pass |
| Corruption robustness | noise 94.27 -> 94.26, brightness 95.93 -> 95.87 | pass |
| Multi-layer replacement | stage1+stage2 TCSLU 95.70 -> 95.43; static hard only 88.99 | pass |
| Support threshold | min support 1/2/4 all give TCSLU 95.81 with 2048 entries | pass |

Interpretation: these rows strengthen the TCSLU route but remain scoped to
QKFormer CIFAR-10 T=4 current replacement. They should not be described as full
model replacement, measured hardware acceleration, or cross-architecture
generalization.

Paper status: the QKFormer CIFAR-10/100 T=1/T=4 main current-replacement rows,
the CIFAR-10 T=4 semantic controls, and the two-target/robustness summary are
included in `paper/sections/4_experiments.tex`. Cross-architecture
generalization remains a separately reserved future subsection/table.

### Completed Priority-1 Lookup Cost Proxy

The first post-TCSLU priority experiment was completed as a metadata-inclusive
analytical lookup-cost proxy, not a measured hardware result:

- Table: `paper/tables/lookup_cost_proxy.md`
- CSV/JSON: `paper/tables/data/lookup_cost_proxy.csv` and
  `paper/tables/data/lookup_cost_proxy.json`
- Script: `scripts/local/build_lookup_cost_proxy_tables.py`

Key findings:

- E7 `TC+QK` grows from 55.87 KiB prototype-only FP32 values to 141.42 KiB
  under sparse FP32 values + uint32 keys + uint16 counters + valid bits.
- Dense compact-full-address FP32 value storage for the E7 canonical address
  space is 280.50 KiB including a valid bitset, so metadata-aware `TC+QK`
  remains about 1.98x smaller than dense full-address storage.
- QKFormer TCSLU current replacement uses a 2048-entry address space; sparse
  key-value metadata is wasteful there (20.25 KiB) compared with a dense FP32
  value table plus valid bitset (8.25 KiB). This supports using dense storage
  for the current-unit table and sparse accounting for larger E7 subspace
  tables.
- The QKFormer current-unit lookup-count proxy is 0.010417 of the replaced
  projection MAC count, but this remains an operation-count proxy and excludes
  address generation, memory hierarchy, and measured cycles.

### Completed All-Attention Projection LUT Replacement

All four QKFormer attention projection currents were replaced simultaneously on
CIFAR-100 seed-42 checkpoints:

- `stage1.0.tssa`
- `stage2.0.tssa`
- `stage3.0.ssa`
- `stage3.1.ssa`

The scalar response-prototype route does not scale to all targets: calibrated
prototype lookup gives 56.28% from 77.78% clean at `T=1`, and 54.08% from
81.09% clean at `T=4`.

An exact grouped binary-pattern projection LUT resolves this boundary. The
compact 2-bit-group configuration gives:

| Setting | Clean Acc@1 | Grouped LUT Acc@1 | Delta | FP32 values |
|---|---:|---:|---:|---:|
| CIFAR-100 T=1 | 77.78 | 77.72 | -0.06 | 681,984 |
| CIFAR-100 T=4 | 81.09 | 81.09 | 0.00 | 681,984 |

FP32 value storage is 2,664 KiB; FP16 is 1,332 KiB. Aggregate current MSE is
about `2.10e-9` at `T=1` and `1.82e-9` at `T=4`. The implementation decomposes
each binary-input 1x1 projection into grouped pattern-to-output-contribution
tables and sums the looked-up vectors.

Detailed memo:
`results/qk_grouped_projection_c100_all4_20260618.md`.

Claim boundary: this completes all-attention `proj_conv` current replacement,
not complete-network replacement. Q/K/V projections, MLPs, patch embeddings,
BN/LIF dynamics, and the classifier remain unchanged. Cost numbers are
analytical proxies, not measured speed, SRAM, or energy.

AAAI positioning: treat this row as strong completeness/fidelity support for
the semantic-LUT paper, not as a standalone novelty claim. The grouped
binary-pattern decomposition is algebraically exact, has clear overlap with
prior grouped/truth-table LUT ideas, uses more table values than the frozen
projection weights, and is currently tested through a hook that still executes
the original convolution.

The operator-scope extension was audited under a preregistered stop rule. Do
not call the present result a complete QKFormer, full-attention, or
all-operator LUT replacement.

### All-Q/K/V Fidelity Audit

The preregistered all-affine extension stopped at its first category. All ten
Q/K/V projections were replaced simultaneously on QKFormer CIFAR-100 `T=1`,
seed 42, full validation:

| Attempt | Clean | LUT | Drop | Output NRMSE | Clip | Gate |
|---|---:|---:|---:|---:|---:|---|
| integer levels 8 | 77.58 | 77.13 | 0.45 | 0.00012029 | 0 | PASS |
| integer levels 16 retry | 77.58 | 77.13 | 0.45 | 0.00012029 | 0 | PASS; no benefit |

All observed Q/K/V inputs were exact integers and already lay within `[0, 4]`,
so the retry changed only table size, not predictions. The 0.45-point paired
drop passes the final project-level 0.50-point near-lossless criterion. The
small residual difference is consistent with simultaneous BN/LIF threshold
sensitivity to changed floating-point accumulation order. See
`results/qk_all_affine_qkv_boundary_20260619.md`.

### All-Affine Continuation

The final uniform project criterion is a maximum paired loss of 0.50 pp.

| Category | Targets | Clean | LUT | Drop | NRMSE | FP32 values | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Q/K/V | 10 | 77.58 | 77.13 | 0.45 | 0.00012029 | 30,528 KiB | pass |
| MLP | 8 | 77.58 | 77.63 | -0.05 | 0.00011311 | 85,248 KiB | pass |
| patch embedding | 9 | 77.58 | 77.47 | 0.11 | 0.00012767 | 83,376 KiB | pass |
| classifier | 1 | 77.58 | 77.59 | -0.01 | 0.00339480 | 38,400 KiB | pass |
| cumulative all-affine | 32 | 77.58 | 77.98 | -0.40 | 0.00013521 | 248,208 KiB | pass |

Positive deltas are no-loss fluctuations, not accuracy gains. The cumulative
row replaces every Conv1d, Conv2d, and Linear output in the evaluated
QKFormer CIFAR-100 `T=1` checkpoint. It leaves BN/LIF, pooling, residual
addition, and SSA matrix products unchanged, and the hooks still execute the
original operators.

The 248,208-KiB FP32 value footprint is approximately 242.4 MiB and 9.47 times
the original weight-value count. This supports operator-substitution fidelity,
not compactness, latency, energy, SRAM, or hardware acceleration. See
`results/qk_all_affine_continuation_20260619.md`.

## Raw Evidence Policy

Expanded local JSON files were moved into a compressed cold archive to keep the
default result tree Markdown-first. The archive is not a paper result and need
not be read during normal continuation, but it may be restored for numerical
audits. Existing seed-43/44 evidence is not prohibited; the policy is to avoid
new nonessential seed-43/44 experiments.
