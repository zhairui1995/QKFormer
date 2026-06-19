# QK-LUTFormer Next-Session Handoff

Updated: 2026-06-18

## Start Here

Read, in order:

1. `/Users/cvue/.codex/memory/personal-codex-rules.md`
2. `AGENTS.md`
3. `results/EXPERIMENT_RESULTS_SUMMARY.md`
4. `docs/QK_LUTFORMER_QUICK_STATE.md`
5. this file

Branch: `codex/qkformer-lut-5way-controls`

The default continuation path is Markdown-first. Do not recursively inspect
`results/`, restore cold archives, or run broad JSON analyzers unless a specific
claim needs forensic verification.

## Scientific State

- Verdict: scoped `GO-METHOD`.
- Core method evidence: E1 address reconstruction, E5 explicit hierarchical
  backoff, and E7 compact subspace-decoupled lookup.
- E7 is the main method contribution: aligned semantic subtables preserve most
  full-address reconstruction gain below the 25% entry-count gate.
- Classification accuracy is secondary evidence. The completed CIFAR-100 T=4
  matched-control study supports stable improvement across independently
  trained seed-42/43/44 QKFormer backbones: all five registered gates pass and
  aligned Q/K LUT is best in every gate.
- This classification result strengthens the paper but does not define the
  scoped method verdict. It is not cross-architecture, cross-dataset, ImageNet,
  latency, or energy evidence.
- Existing seed-43/44 checkpoints, metrics, reports, and gate artifacts remain
  valid and may be read, downloaded, audited, summarized, or archived.

## Seed Policy

- All new paper-mainline analysis uses seed 42.
- Existing multi-seed E1/E7 evidence remains valid and may be cited as completed
  robustness evidence.
- Do not launch new seed-43/44 experiments unless a future reviewer request
  makes a specific replication indispensable.
- Do not continue alpha, stage, checkpoint, or random-seed searches.

This is a workflow simplification, not a claim that stochastic variation no
longer exists. The manuscript must distinguish existing robustness evidence
from new seed-42-only analysis.

## Canonical E7 Cell

CIFAR-100 T=1, seed 42, 128 calibration batches, all 313 validation batches,
minimum support count 2:

| Estimator | Reduction | Entries | Entry fraction | Retention | FP32 values |
|---|---:|---:|---:|---:|---:|
| token/channel | 3.4253% | 2,052 | 2.95% | 77.42% | 8.0 KiB |
| TC+Q/gate | 3.8749% | 6,148 | 8.83% | 87.58% | 24.0 KiB |
| TC+QK | 4.1727% | 14,303 | 20.54% | 94.31% | 55.9 KiB |
| full address | 4.4243% | 19,258 | 27.66% | 100.00% | 75.2 KiB |

Shuffled TC+Q/gate and TC+QK controls are -1.8908% and -1.6600%.

Entry fractions use the compact full-address space of 69,632 entries. The FP32
footprint is an idealized prototype-value array with four global fallback
scalars; it is not measured SRAM and excludes keys, indices, counters, padding,
allocator overhead, and hardware metadata.

Traceable paper snapshot:

```text
paper/figures/data/fig5_e7_component_ablation_data.json
```

## Paper Workflow

1. Keep the title and contribution centered on compact, auditable Q/K lookup
   addresses and subspace reconstruction.
2. Use existing multi-seed E1/E7 rows as completed robustness evidence.
3. Use the canonical seed-42 E7 cell for the new component ablation, entry
   budget, gain-retention, and idealized footprint analysis.
4. Include the completed five-gate classification result as a secondary,
   protocol-scoped finding. State explicitly that the replicated backbones are
   independently trained checkpoints of the same architecture and dataset.
5. Keep explicit limitations on SRAM, latency, energy, ImageNet, production
   acceleration, and complete QKFormer replacement.
6. Compile and audit the LaTeX after every claim-level revision.

Generate Figure 5:

```bash
python3 scripts/plot_qk_lut_e7_component_ablation.py
```

Compile:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Required checks:

- no undefined references or citations;
- no overfull boxes;
- page count recorded;
- every new number maps to the Figure 5 JSON snapshot;
- historical seed-43/44 results remain traceable and are clearly separated from
  new seed-42-only analysis;
- every classification row maps to the authoritative result summary or an
  audited gate artifact;
- no manuscript sentence still labels the completed classification study as
  `PENDING`.

## Current Paper Bottlenecks

1. **Method-to-system gap:** E7 demonstrates compactness with supported-entry
   counts and idealized FP32 prototype values, not measured SRAM, latency,
   energy, area, or real sparse-index overhead.
2. **Generality gap:** classification stability is replicated across training
   seeds of one QKFormer/CIFAR-100 T=4 setting, not across architectures,
   datasets, or ImageNet-scale workloads.
3. **Remaining replacement gap:** all four attention projection currents now
   have a near-lossless grouped-LUT result, but Q/K/V projections, MLPs,
   patch embeddings, BN/LIF dynamics, and complete attention blocks are not
   replaced.
4. **Story-density risk:** E1, E5, E7, and the classification gate can read as
   separate studies unless the manuscript makes the causal chain explicit:
   address semantics -> transparent misses -> compact decomposition -> bounded
   downstream utility.
5. **Novelty-positioning risk:** the paper must distinguish semantic Q/K spike
   addressing and auditable subtable composition from generic quantization,
   hashing, associative memory, prototype lookup, and hardware LUT mapping.

## Recent Replacement-Fidelity Boundary

Current conclusion file:
`docs/QKFORMER_SPIKFORMER_DIFFERENCE_CONCLUSION.md`.

- QKFormer CIFAR-10 T=1 current-unit probe: full-validation moment/TCSLU hard
  current LUT on `stage1.0.tssa` gives 94.64% clean vs 94.57% injected Acc@1.
- QKFormer CIFAR-10 T=4 current-unit probe: full-validation moment/TCSLU hard
  current LUT on `stage1.0.tssa` gives 95.70% clean vs 95.81% injected Acc@1.
  Treat the positive delta as no-loss fluctuation, not as an accuracy-gain
  claim.
- Spikformer-4-384w CIFAR-10 T=4 diagnostics were archived at
  `results/spikformer_experiment_archive_20260617.zip`. The only active
  retained row is learned temporal+channel gate: 88.16% clean vs 76.28%
  injected Acc@1.
- Interpretation: the present lookup-current design fits QKFormer T=1/T=4 in
  the single-layer current diagnostic. Spikformer T=4 remains a boundary case
  because T>1 BN/LIF temporal state and cross-channel current geometry amplify
  LUT mismatch. Treat Spikformer as discussion evidence, not a second positive
  backbone.

## TCSLU AAAI2027 Track

Use `docs/TCSLU_AAAI2027_EXPERIMENT_PLAN.md` as the controlling plan for the
next experimental paper line.

- Method: Temporal-Channel Calibrated Semantic Lookup Unit (TCSLU).
- Main claim to test: semantic lookup needs temporal/channel calibration to
  preserve spiking Transformer dynamics.
- Existing CIFAR-100 E1/E5/E7 evidence is already counted; do not repeat it.
- New CIFAR-100 work should be current replacement and TCSLU ablations.
- Peer context data is in
  `paper/tables/data/tcslu_peer_spiking_transformer_comparison.csv`.
- Generate the current planning tables with:

```bash
python3 scripts/local/build_tcslu_aaai_tables.py
```

Remote entrypoint:

```bash
bash scripts/server/run_tcslu_aaai_matrix.sh 0 qkf_c100_t1_full
```

Only promote planned rows to paper results after full-validation runs produce
machine-readable CSV/JSON and pass the preregistered gates.

Current promoted rows in `paper/tables/data/tcslu_current_replacement_status.csv`
still cover the earlier CIFAR-100 and Spikformer rows:

- QKFormer CIFAR-100 T=1 result dir:
  `results/tcslu_qkf_c100_t1_full_20260618_104451`.
- QKFormer CIFAR-100 T=4 result dir:
  `results/tcslu_qkf_c100_t4_full_20260618_104512`.
- Spikformer-4-384w CIFAR-10 T=4 result dir:
  `results/tcslu_spik_c10_t4_boundary_rerun_20260618_105951`.

The completed QKFormer CIFAR-10/100 full-validation rows are now promoted to
the end-to-end current-replacement subsection in
`paper/sections/4_experiments.tex`:

- QKFormer CIFAR-10 T=1 result dir:
  `results/tcslu_qkf_c10_t1_full_20260618_161423`.
- QKFormer CIFAR-10 T=4 result dir:
  `results/tcslu_qkf_c10_t4_full_20260618_161423_t4rerun`.

Key finding: QKFormer CIFAR-10 T=1/T=4 and CIFAR-100 T=1/T=4 all support the
local current-replacement route. Static hard LUT is not always sufficient, but
moment/channel/TCSLU-style calibration keeps QKFormer replacement near-lossless
in the completed single-layer diagnostics. The manuscript reserves
`sec:cross_arch_generalization` and `tab:cross_arch_generalization` for a
future cross-architecture result; do not merge that evidence into the
QKFormer-only table. Spikformer remains a boundary case.

Additional CIFAR-10 shuffled-address controls are complete:

- T=1 result dir:
  `results/tcslu_qkf_c10_t1_shuffled_20260618_163112`.
- T=4 result dir:
  `results/tcslu_qkf_c10_t4_shuffled_20260618_163112`.
- Summary memo:
  `results/tcslu_qkf_c10_shuffled_control_20260618.md`.

Headline: aligned Q/K lookup beats shuffled on both T=1 and T=4. The T=4
control is especially strong: static hard 95.08 vs 90.57, and TCSLU/moment
95.81 vs 95.31.

The full requested follow-up queue is now complete. Summary memo:
`results/tcslu_qkf_c10_t4_followup_experiments_20260618.md`.

- Semantic Table C controls: aligned TCSLU 95.81 is best among global 95.63,
  random 95.52, token/channel 95.45, and shuffled 95.31.
- Calibration-subset replication (five fixed subsets) gives aligned 95.820,
  global 95.630, and token/channel 95.542 mean Acc@1. Aligned is higher in all
  five subsets. Aligned minus global and token/channel are +0.190 pp and
  +0.278 pp, with 95% CIs [+0.037, +0.343] and [+0.014, +0.542]; both exclude
  zero. Global minus token/channel is +0.088 pp with 95% CI
  [-0.045, +0.221] and is not a stable ordering.
- Corruption robustness: noise and brightness severity 1 are near-lossless.
- Multi-layer replacement: `stage1.0.tssa,stage2.0.tssa` TCSLU gives
  95.70 -> 95.43; static hard gives 88.99.
- Support threshold 1/2/4: all match the support-2 baseline at TCSLU 95.81.

Completion report:
`docs/TCSLU_AAAI2027_COMPLETED_REPORT.md`.

### Completed Priority-1 Cost Accounting

The first post-TCSLU priority experiment is now complete: metadata-inclusive
lookup-cost accounting. It is analytical only, but it closes the immediate
paper hygiene gap where earlier footprint numbers counted prototype values
without keys, counters, valid bits, or index metadata.

Outputs:

- `paper/tables/lookup_cost_proxy.md`
- `paper/tables/data/lookup_cost_proxy.csv`
- `paper/tables/data/lookup_cost_proxy.json`
- `scripts/local/build_lookup_cost_proxy_tables.py`

Headline numbers:

- E7 `TC+QK`: 55.87 KiB prototype-only, 141.42 KiB sparse FP32 with uint32
  key + uint16 counter + valid bit, versus 280.50 KiB dense compact-full-
  address FP32 storage.
- QKFormer current TCSLU: dense 2048-entry table is better than sparse
  key-value storage, 8.25 KiB dense FP32 with valid bit versus 20.25 KiB sparse
  FP32 metadata-inclusive.
- QKFormer current-unit lookup-count proxy is 0.010417 of the replaced
  projection MAC count, but this is not latency, SRAM, energy, or measured
  hardware-cycle evidence.

### Completed All-Attention Projection Replacement

The prior reconstruction-to-replacement bottleneck is now narrowed further.
All four attention `proj_conv` currents were replaced simultaneously on
CIFAR-100:

- prototype TCSLU boundary:
  - T=1: 77.78 -> 56.28;
  - T=4: 81.09 -> 54.08;
- most compact tested grouped row, using 2-bit binary patterns:
  - T=1: 77.78 -> 77.72;
  - T=4: 81.09 -> 81.09;
  - FP32 value footprint: 2,664 KiB;
  - aggregate current MSE: approximately `2e-9`.

This result covers `stage1.0.tssa`, `stage2.0.tssa`, `stage3.0.ssa`, and
`stage3.1.ssa`. It is all-attention projection-current replacement, not
complete-network LUT replacement. See
`results/qk_grouped_projection_c100_all4_20260618.md`.

## Novelty Position For AAAI Review

Treat the all-attention result as strong completeness/fidelity support, not as
the paper's standalone core novelty:

- it closes the single-layer objection by replacing all four attention
  projection currents simultaneously;
- it is an exact algebraic decomposition of frozen binary-input projections,
  so novelty overlap with grouped lookup, truth-table, LUTNet, LogicNets, and
  PolyLUT-style methods must be addressed;
- the 2-bit table uses 2,664 KiB of FP32 values and therefore does not establish
  storage compression;
- the current hook still executes the original `proj_conv`, so the result does
  not establish runtime removal or hardware acceleration.

The recommended contribution hierarchy is:

1. semantic Q/K lookup addressability;
2. support-aware backoff and compact subspace lookup;
3. calibrated current-replacement fidelity;
4. all-attention grouped projection replacement as the completeness closure.

Detailed review handoff and a copy-ready ChatGPT Pro prompt:
`docs/AAAI_ALL_ATTENTION_REVIEW_HANDOFF_ZH.md`.

## Next Work

The operator-scope extension is complete under the final uniform 0.50-pp
near-lossless criterion. Do not start a new accumulator implementation,
post-LUT calibration, bit-width sweep, checkpoint replication, or hardware
claim experiment by default.

Keep “all-attention projection-current replacement” as the compact registered
completeness claim. The later row may be described explicitly as exploratory
“all Conv1d/Conv2d/Linear output substitution” under the final 0.50-pp
criterion. Never call it “complete QKFormer,” “full attention,” or
“all-operator” LUT replacement.

## Session Completion And SSH Note

- All formal jobs completed and generated complete machine-readable metrics.
- A final SSH status recheck timed out once after completion.
- The timeout is an operational verification note, not evidence of an
  unfinished experiment. Do not rerun formal jobs solely because of it.

## All-Affine Replacement Fidelity

The planned Q/K/V -> MLP -> patch -> classifier -> cumulative audit was
preregistered in `docs/QKFORMER_ALL_AFFINE_LUT_PREREGISTRATION.md`.

Q/K/V was tested with the main configuration and one range retry:

- main integer range `[0, 7]`: 77.58 -> 77.13, drop 0.45 pp;
- retry integer range `[0, 15]`: identical 77.58 -> 77.13;
- aggregate output NRMSE 0.00012029 and zero clipping;
- all ten Q/K/V targets executed;
- final project maximum drop: 0.50 pp.

Observed inputs were exact integers and no larger than four, so the retry only
doubled table storage. This rules out address range as the failure cause. The
working hypothesis is BN/LIF threshold amplification of tiny accumulation-order
error.

Q/K/V passes. MLP, patch, classifier, and cumulative all-affine also pass:

- MLP: 77.58 -> 77.63;
- patch: 77.58 -> 77.47;
- classifier: 77.58 -> 77.59;
- all 32 Conv1d/Conv2d/Linear modules: 77.58 -> 77.98.

The cumulative result supports scalar-level LUT-equivalent substitution
fidelity for all convolutional and linear operators in this single
QKFormer/CIFAR-100/T=1 checkpoint. It is not complete QKFormer replacement:
BN/LIF, pooling, residual addition, and SSA matrix products remain. FP32 LUT
values total 248,208 KiB (9.47x the original weight-value count), and the hooks
still execute the original modules.

Detailed evidence:
`results/qk_all_affine_continuation_20260619.md`.

## Hygiene

- Preserve unrelated dirty and untracked files.
- Do not commit datasets, checkpoints, logs, tarballs, or credentials.
- Do not store private authentication material in scripts or documentation.
- Keep generated experiment outputs on the server and paper-ready evidence in
  small traceable Markdown/CSV/JSON snapshots.
