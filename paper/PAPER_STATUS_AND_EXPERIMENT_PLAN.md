# QK-LUTFormer Paper Status And Experiment Plan

Status date: 2026-06-18

## Editorial Decision

Scoped `GO-METHOD`.

The manuscript presents a compact response-reconstruction method built from:

1. address-specific held-out reconstruction (E1);
2. explicit support-aware hierarchical backoff (E5);
3. subspace-decoupled Q/K lookup below a 25% entry-count gate (E7).

Classification is not a core contribution. The completed CIFAR-100 T=4
matched-control study passes all five registered one-/two-epoch gates across
the independently trained seed-42/43/44 QKFormer checkpoints. This is
protocol-scoped secondary evidence, not cross-architecture or cross-dataset
generalization.

## Evidence Included

### E1

- CIFAR-10 T=4: address 3.7080%, token/channel 2.7941%, shuffled -17.8260%.
- CIFAR-100 T=1: address 4.4247%, token/channel 3.4246%, shuffled -15.8216%.
- Existing checkpoint and calibration-subset results preserve the
  address-over-token/channel ordering.

### E5

- Fallback levels are explicit and sum to all validation lookups.
- From eight calibration batches onward, hierarchy beats token/channel across
  the completed grid.
- The 40.4%--51.1% effective-entry range misses the compactness target, so E5
  is presented as transparent miss handling rather than the final compact
  design.

### E7

Existing multi-seed/calibration-size results establish stable ordering. The new
paper analysis uses one fixed cell: CIFAR-100 T=1, seed 42, 128 calibration
batches, full validation.

| Estimator | Reduction | Entries | Entry fraction | Retention | FP32 values |
|---|---:|---:|---:|---:|---:|
| token/channel | 3.4253% | 2,052 | 2.95% | 77.42% | 8.0 KiB |
| TC+Q/gate | 3.8749% | 6,148 | 8.83% | 87.58% | 24.0 KiB |
| TC+QK | 4.1727% | 14,303 | 20.54% | 94.31% | 55.9 KiB |
| full address | 4.4243% | 19,258 | 27.66% | 100.00% | 75.2 KiB |

Shuffled TC+Q/gate and TC+QK controls are -1.8908% and -1.6600%. FP32 values
are an idealized prototype-only footprint, not measured SRAM.

## Experiment Policy

- Do not start new training before writing.
- New paper-mainline analysis uses seed 42.
- Existing multi-seed E1/E7 evidence remains admissible as completed
  robustness evidence.
- Do not run new seed-43/44, alpha, stage, checkpoint, or random-seed searches.
- Existing seed-43/44 evidence remains part of the project record and may be
  audited or summarized; do not launch new seed-43/44 experiments by default.
- A future cross-backbone classification study requires a new explicit
  pre-registration and is outside the default loop.

## Writing Plan

1. Lead with E1 address semantics and shuffled controls.
2. Use E5 to establish explicit miss handling and motivate E7.
3. Present E7 as the method contribution, with Figure 5 and the canonical
   seed-42 table.
4. Keep classification as secondary, protocol-scoped evidence rather than the
   basis of the method verdict.
5. Separate entry/FP32 proxies from measured hardware claims.
6. Present all-attention projection replacement as completeness/fidelity
   support, not as a standalone novelty center.
7. Keep the all-affine continuation exploratory and disclose its final
   0.50-point criterion and 9.47x table-value expansion.
8. Compile, check page count and layout, then run citation and numerical claim
   traceability audits.

## Claim Boundary

Supported:

- structured Q/K lookup addresses;
- address-specific held-out reconstruction;
- calibration efficiency;
- explicit hierarchical fallback;
- compact subspace reconstruction under an entry-count proxy.

Secondary evidence:

- stable aligned-Q/K classification ordering across five registered
  same-architecture CIFAR-100 T=4 gates.

Not supported:

- measured SRAM, latency, area, energy, or hardware speedup;
- ImageNet generalization;
- production LUT acceleration;
- complete QKFormer replacement.

## All-Attention Projection Replacement

All four attention `proj_conv` currents are replaced simultaneously by a
grouped binary-pattern LUT on CIFAR-100 seed-42 `T=1` and `T=4`. The compact
2-bit row gives 77.78 -> 77.72 and 81.09 -> 81.09 Acc@1, respectively.

This is a strong completeness/fidelity result, but not the paper's independent
novelty center: it is an exact grouped decomposition, uses 2,664 KiB of FP32
table values, and the current hook still executes the original projection.
The later operator-scope audit covers all 32 Conv1d/Conv2d/Linear modules in
the evaluated CIFAR-100 `T=1` checkpoint. Under the final uniform 0.50-point
near-lossless criterion, Q/K/V, MLP, patch embedding, classifier, and
cumulative all-affine rows pass; the cumulative row is 77.58 -> 77.98.

This is exploratory substitution-fidelity evidence, not a compact design:
FP32 table values total 248,208 KiB (9.47x the original weight-value count).
BN/LIF, pooling, residual addition, and SSA matrix products remain unchanged.

Report:
`../results/qk_all_affine_continuation_20260619.md`.

Review handoff:
`../docs/AAAI_ALL_ATTENTION_REVIEW_HANDOFF_ZH.md`.

## Reproducible Artifacts

- Figure data: `figures/data/fig5_e7_component_ablation_data.json`
- Figure command: `python3 ../scripts/plot_qk_lut_e7_component_ablation.py`
- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
