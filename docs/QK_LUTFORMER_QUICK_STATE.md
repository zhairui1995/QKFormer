# QK-LUTFormer Quick State

Updated: 2026-06-18

Purpose: low-token bootstrap. Read
`results/EXPERIMENT_RESULTS_SUMMARY.md` first, then this file. Open historical
handoffs or raw metrics only when implementation history or a disputed number
requires them.

## Current Verdict

Scoped `GO-METHOD` for:

1. Q/K-address-specific held-out response reconstruction (E1)
2. explicit support-aware hierarchical backoff (E5)
3. compact subspace-decoupled Q/K lookup (E7)

Classification accuracy is not the paper's core contribution, but the completed
CIFAR-100 T=4 matched-control study supports stable improvement across the
independently trained seed-42/43/44 QKFormer checkpoints. All five registered one-/two-
epoch gates pass, and aligned Q/K LUT has the highest mean Acc@1 in every gate.
This is protocol-scoped downstream evidence, not cross-architecture or
cross-dataset generalization. Existing seed-43/44 results remain valid; the
default policy is to avoid launching new seed-43/44 experiments.

## Main Evidence

### E1: Address Semantics

- CIFAR-100 T=1, 128 calibration batches: address 4.4247%, token/channel
  3.4246%, shuffled -15.8216% relative MSE reduction.
- CIFAR-10 T=4: address 3.7080%, token/channel 2.7941%, shuffled -17.8260%.
- Existing CIFAR-10 T=1 checkpoints 42/43/44 preserve address over
  token/channel ordering.
- Existing calibration-size runs show that the ordering holds from 8 to 512
  batches.

Interpretation: fine Q/K address alignment predicts held-out local responses
beyond coarse state and table capacity.

### E5: Explicit Backoff

- From eight calibration batches onward, support-aware hierarchy beats the
  token/channel baseline across the completed grid.
- Every validation lookup records its fallback level; misses are not hidden.
- Effective entries are 40.4%--51.1% of the compact full-address space, so E5
  is a partial compactness result.

### E7: Compact Subspace LUT

- `TC+Q/gate` and `TC+QK` beat token/channel and matched shuffled-subspace
  controls throughout the completed grid.
- `TC+Q/gate` uses 8.83% of the compact full-address address-space entry count.
- `TC+QK` uses about 20.2%--20.6%, below the 25% entry gate.

Canonical paper-analysis cell: CIFAR-100 T=1, seed 42, 128 calibration batches,
full validation.

| Estimator | Reduction | Entries | Entry fraction | Retention | FP32 values |
|---|---:|---:|---:|---:|---:|
| token/channel | 3.4253% | 2,052 | 2.95% | 77.42% | 8.0 KiB |
| TC+Q/gate | 3.8749% | 6,148 | 8.83% | 87.58% | 24.0 KiB |
| TC+QK | 4.1727% | 14,303 | 20.54% | 94.31% | 55.9 KiB |
| full address | 4.4243% | 19,258 | 27.66% | 100.00% | 75.2 KiB |

Shuffled TC+Q/gate and TC+QK reductions are -1.8908% and -1.6600%. The FP32
column is an idealized value-array proxy only; it excludes lookup keys, indices,
counters, alignment, allocator overhead, and hardware metadata.

### Secondary Classification Evidence

| Backbone / budget | Aligned | Shuffled | Global | Baseline | Delta |
|---|---:|---:|---:|---:|---:|
| seed 42 / 1 epoch | 81.73% | 81.62% | 81.46% | 81.22% | +0.51 |
| seed 42 / 2 epochs | 81.75% | 81.28% | 81.23% | 81.22% | +0.53 |
| seed 43 / 1 epoch | 81.57% | 81.37% | 81.36% | 81.37% | +0.20 |
| seed 44 / 1 epoch | 81.41% | 81.24% | 81.29% | 81.04% | +0.37 |
| seed 44 / 2 epochs | 81.61% | 81.39% | 81.34% | 81.04% | +0.57 |

Interpretation: aligned Q/K lookup is consistently best under the registered
CIFAR-100 T=4 protocol. Keep the exact scope visible: independent checkpoints
of one architecture and dataset, not broad model-family generalization.

## Default Workflow

1. Use seed 42 for all new paper-mainline analysis.
2. Reuse existing multi-seed E1/E7 results only as already-completed robustness
   evidence.
3. Preserve existing seed-43/44 evidence, but do not run new seed-43/44
   experiments unless a concrete reviewer request requires replication.
4. Do not continue alpha, stage, checkpoint, or seed searches.
5. Prioritize Figure 5, main tables, classification-table synchronization,
   claim traceability, manuscript revision, and LaTeX verification.
6. Do not launch another experiment until it names one unresolved paper claim,
   a matched control, a fixed budget, and pass/fail criteria.
7. After the paper audit, choose at most one next evidence track: measured
   system cost, cross-architecture/cross-dataset generalization, or end-to-end
   replacement fidelity.

Current figure command:

```bash
python3 scripts/plot_qk_lut_e7_component_ablation.py
```

Compile the paper:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Claim Boundary

Allowed:

- structured Q/K lookup addresses;
- held-out address-specific reconstruction;
- calibration efficiency;
- explicit backoff and miss accounting;
- compact subspace reconstruction under an entry-count proxy;
- stable CIFAR-100 T=4 classification improvement across independently trained
  seed-42/43/44 QKFormer checkpoints under the registered matched-control protocol.

Not allowed:

- measured SRAM, latency, energy, or hardware speedup;
- ImageNet generalization;
- cross-architecture or cross-dataset classification generalization;
- production LUT acceleration;
- a complete LUT replacement for QKFormer.

## Current Replacement-Fidelity Boundary

Recent hook diagnostics should remain scoped separately from the paper-core E1,
E5, and E7 evidence.

- QKFormer CIFAR-10 T=1, `stage1.0.tssa`, current before original BN/LIF:
  full-validation moment/TCSLU hard LUT gives 94.64% -> 94.57% Acc@1.
- QKFormer CIFAR-10 T=4, same target and protocol: full-validation
  moment/TCSLU hard LUT gives 95.70% -> 95.81% Acc@1. Treat the positive delta
  as no-loss fluctuation, not as an accuracy-gain claim.
- Spikformer-4-384w CIFAR-10 T=4: the only active retained row is learned
  temporal+channel gate, 88.16% -> 76.28% Acc@1. Detailed Spikformer
  diagnostics are archived at `results/spikformer_experiment_archive_20260617.zip`.

Conclusion file:
`docs/QKFORMER_SPIKFORMER_DIFFERENCE_CONCLUSION.md`. Current interpretation:
QKFormer T=1 and T=4 match the semantic lookup-current contract in this
single-layer diagnostic; Spikformer T=4 exposes an unsolved temporal/channel
dynamics mismatch and should be treated as a boundary discussion, not a
parallel positive mainline.

## TCSLU AAAI2027 Track

The next paper line is preregistered in
`docs/TCSLU_AAAI2027_EXPERIMENT_PLAN.md`.

Method name: **TCSLU**, Temporal-Channel Calibrated Semantic Lookup Unit.

Research question: static semantic LUT is near-lossless in QKFormer T=1 but
fails under Spikformer T=4; test whether semantic address lookup plus
moment-matching, temporal gates, channel calibration, and support-aware backoff
can recover the missing dynamic contract.

Machine-readable assets:

- `paper/tables/data/tcslu_experiment_registry.csv`
- `paper/tables/data/tcslu_current_replacement_status.csv`
- `paper/tables/data/tcslu_peer_spiking_transformer_comparison.csv`
- `scripts/local/build_tcslu_aaai_tables.py`
- `scripts/server/run_tcslu_aaai_matrix.sh`

Default next run is not another E1/E7 sweep. It is QKFormer CIFAR-100 current
replacement/TCSLU ablation, with Spikformer kept as the temporal-channel
mismatch recovery target.

Completed current-replacement main rows:

- QKFormer CIFAR-10 T=1: static hard 94.64 -> 93.64; TCSLU/moment hard
  94.64 -> 94.57.
- QKFormer CIFAR-10 T=4: static hard 95.70 -> 95.08; TCSLU/moment hard
  95.70 -> 95.81. Do not claim accuracy gain from the positive delta.
- QKFormer CIFAR-100 T=1: static hard 77.78 -> 75.55; TCSLU/moment hard
  77.78 -> 77.26.
- QKFormer CIFAR-100 T=4: static hard 81.09 -> 68.02; TCSLU/moment hard
  81.09 -> 80.65.
- Spikformer-4-384w CIFAR-10 T=4: static hard 88.16 -> 66.47; learned
  temporal+channel gate 88.16 -> 76.35.

Key finding: QKFormer CIFAR-10 T=1/T=4 and CIFAR-100 T=1/T=4 now support the
same local current-replacement story: Q/K spike-address LUTs can query local
responses, and moment/channel/TCSLU-style calibration preserves the dynamic
contract. Spikformer remains the boundary case.

CIFAR-10 shuffled-address controls are now complete under the same
full-validation protocol:

- T=1 static aligned vs shuffled: 93.64 vs 91.48; TCSLU/moment aligned vs
  shuffled: 94.57 vs 94.37.
- T=4 static aligned vs shuffled: 95.08 vs 90.57; TCSLU/moment aligned vs
  shuffled: 95.81 vs 95.31.

Interpretation: preserving Q/K spike-address alignment improves replacement
fidelity beyond distribution-matched prototype smoothing.

Additional QKFormer CIFAR-10 T=4 follow-up groups are complete:

- Semantic controls: aligned TCSLU 95.81 beats global 95.63, random 95.52,
  token/channel 95.45, and shuffled 95.31.
- Five fixed calibration subsets refine this result: aligned averages 95.820,
  global 95.630, and token/channel 95.542. Aligned exceeds both controls in all
  five subsets. Aligned minus global and token/channel are +0.190 pp and
  +0.278 pp, with 95% CIs [+0.037, +0.343] and [+0.014, +0.542]; both exclude
  zero. Global minus token/channel is only +0.088 pp with 95% CI
  [-0.045, +0.221], so those two controls are not distinguishable.
- Corruption robustness: noise severity 1 is near-lossless, 94.27 -> 94.26;
  brightness severity 1 is near-lossless, 95.93 -> 95.87.
- Multi-layer current replacement: `stage1.0.tssa,stage2.0.tssa` TCSLU gives
  95.70 -> 95.43, while static hard drops to 88.99.
- Support threshold 1/2/4: all give TCSLU 95.81 with 2048 supported entries.

Summary memo:
`results/tcslu_qkf_c10_t4_followup_experiments_20260618.md`.

Traceable table for already promoted paper rows:
`paper/tables/tcslu_aaai2027_tables.md`.

The QKFormer CIFAR-10/100 full-validation current-replacement rows and the
CIFAR-10 T=4 semantic controls are now promoted into
`paper/sections/4_experiments.tex`. The paper keeps them in a QKFormer-only
subsection and reserves separate labels for the future cross-architecture
generalization result.

Completion report:
`docs/TCSLU_AAAI2027_COMPLETED_REPORT.md`.

### Priority-1 Lookup Cost Proxy Completed

The highest-priority next experiment after the TCSLU current-replacement rows
was the system-cost accounting gap: prior KiB values were prototype-only and
excluded keys, counters, valid bits, indexing, and hardware metadata. A
metadata-inclusive analytical proxy is now generated by:

```bash
python3 scripts/local/build_lookup_cost_proxy_tables.py
python3 scripts/local/build_tcslu_aaai_tables.py
```

Outputs:

- `paper/tables/lookup_cost_proxy.md`
- `paper/tables/data/lookup_cost_proxy.csv`
- `paper/tables/data/lookup_cost_proxy.json`

Main interpretation:

- E7 `TC+QK`: 55.87 KiB prototype-only -> 141.42 KiB sparse FP32 with uint32
  key, uint16 counter, and valid bit; still smaller than the 280.50 KiB dense
  compact-full-address FP32 table.
- QKFormer current TCSLU: 2048-address dense storage is preferable; sparse
  key-value accounting would inflate 8.0 KiB prototype values to 20.25 KiB.
- This remains an analytical proxy, not measured SRAM, latency, energy, or
  hardware-cycle evidence.

### All-Attention Projection Replacement Completed

The four attention `proj_conv` currents are now replaced simultaneously on
QKFormer CIFAR-100:

| Group bits | T=1 clean -> LUT | T=4 clean -> LUT | FP32 footprint |
|---:|---:|---:|---:|
| 8 | 77.78 -> 77.73 | 81.09 -> 81.15 | 42,624 KiB |
| 4 | 77.78 -> 77.74 | 81.09 -> 81.11 | 5,328 KiB |
| 2 | 77.78 -> 77.72 | 81.09 -> 81.09 | 2,664 KiB |

The 2-bit grouped binary-pattern LUT is the most compact grouped row tested. It replaces
`stage1.0.tssa`, `stage2.0.tssa`, `stage3.0.ssa`, and `stage3.1.ssa`
projection currents. This closes the all-attention projection replacement gap,
but not complete-network replacement.

Memo: `results/qk_grouped_projection_c100_all4_20260618.md`.

Paper position: this is strong completeness/fidelity evidence, but not the
standalone novelty center. The grouped LUT is an exact algebraic decomposition
of binary-input projections, uses more table values than the original
projection weights, and is currently evaluated through a hook that still
executes the convolution. Keep the core novelty on semantic Q/K addressing,
support-aware backoff, and compact subspace lookup.

The final project-level near-lossless criterion is 0.50 pp. Q/K/V drops by
0.45 pp and is `PASS`.

ChatGPT Pro review handoff:
`docs/AAAI_ALL_ATTENTION_REVIEW_HANDOFF_ZH.md`.

### All-Q/K/V Output Substitution

All Q/K/V projections were substituted simultaneously:

- main 8-level integer LUT: 77.58 -> 77.13, drop 0.45 pp;
- only retry, 16 levels: identical 77.58 -> 77.13;
- output NRMSE: 0.00012029;
- clipping: zero;
- all ten Q/K/V targets executed.

The inputs were already exact integers within `[0, 4]`, so increasing the range
could not improve fidelity. The paired 0.45 pp drop passes the final 0.50 pp
criterion.

Memo: `results/qk_all_affine_qkv_boundary_20260619.md`.

### All-Affine Continuation Completed

All remaining categories pass the uniform amended gate:

| Category | Clean -> LUT | Drop | NRMSE |
|---|---:|---:|---:|
| MLP | 77.58 -> 77.63 | -0.05 | 0.00011311 |
| patch embedding | 77.58 -> 77.47 | 0.11 | 0.00012767 |
| classifier | 77.58 -> 77.59 | -0.01 | 0.00339480 |
| cumulative 32 affine modules | 77.58 -> 77.98 | -0.40 | 0.00013521 |

The cumulative row covers every Conv1d, Conv2d, and Linear module in the
evaluated QKFormer CIFAR-100 `T=1` checkpoint. It does not cover BN/LIF,
pooling, residual addition, or SSA matrix products. Its FP32 LUT values occupy
248,208 KiB (9.47x the original weight-value count), so it is fidelity evidence,
not compactness or measured hardware evidence.

Memo: `results/qk_all_affine_continuation_20260619.md`.
