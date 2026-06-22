# QK-LUTFormer Paper Restructure Review

Date: 2026-06-22

This file records the manuscript changes made when reorganizing the paper
around semantic lookup-addressability and dynamic calibration. The clean
LaTeX manuscript remains submission-readable; review markers are kept here
instead of embedded in the paper.

## Paper Identity

- `[REWRITE]` Title changed to **QK-LUTFormer: Semantic Lookup Addressability
  and Dynamic Calibration for Spiking Attention**.
- `[KEEP]` QK-LUTFormer is the framework name.
- `[ADD]` TCSLU is defined as the dynamic current-calibration module inside
  QK-LUTFormer.
- `[CUT]` Anchor-neuron firing-order encoding is not presented as a method
  because the evaluated implementation uses deterministic mixed-radix Q/K
  fields.
- `[KEEP]` Accuracy is downstream fidelity evidence, not a SOTA objective.

## Abstract

- `[REWRITE]` Replaced the result inventory with a
  problem--method--evidence--boundary structure.
- `[KEEP]` Retained the core E1, E7, TCSLU, and QK-contract numbers.
- `[CUT→SUPP]` Removed grouped projection and native affine/BN/LIF details from
  the abstract.
- `[ADD]` CIFAR10-DVS Route 1 and Route 3 are now completed bounded evidence.
  Route 3 is reported as 81.7% LUT-only from an 81.2% from-scratch teacher,
  not as reproduction of the published 86.7%.
- `[PENDING]` ImageNet-100 and DVS128 LUT placeholders do not appear in the
  abstract.

## Introduction

- `[ADD]` Organized the motivation around three obstacles: semantic validity,
  sparse/large tables, and BN/LIF amplification of current drift.
- `[REWRITE]` Changed broad cross-architecture language to
  contract-conditioned transfer.
- `[REWRITE]` Compressed the contributions to addressability, residual
  backoff/subspaces, TCSLU, and bounded validation.
- `[MOVE]` Operator-completeness results no longer compete with the main
  contribution list.

## Related Work

- `[KEEP]` Spiking attention, LUT inference, compositional memory, and
  conversion/calibration literature.
- `[REWRITE]` The comparison table now contrasts address source, semantic
  controls, miss handling, dynamic calibration, and end-to-end role.
- `[CUT]` Removed the cross-dataset accuracy ranking of unrelated Spiking
  Transformer backbones.

## Method

- `[ADD]` Section 3.1 defines lookup-addressability through occupancy,
  held-out predictiveness, support reliability, and dynamic fidelity.
- `[REWRITE]` Section 3.2 documents the code-faithful mixed-radix addresses.
- `[ADD]` Explicitly states that the method uses neither anchor firing order
  nor raw full-vector truth-table addresses.
- `[ADD]` Distinguishes collision-free tuple packing from quantization aliasing
  and support sparsity.
- `[ADD]` Section 3.3 promotes global-residual estimation and hierarchical
  backoff to a core method with equations.
- `[REWRITE]` Section 3.4 expresses E7 as supported semantic residual subtables.
- `[ADD]` Section 3.5 gives the TCSLU time/channel calibration equation and the
  complete semantic-address-to-BN/LIF flow.
- `[MOVE]` Grouped projection and native operator construction moved to the
  supplementary appendix.

## Experiments

- `[REWRITE]` Reorganized historical E-number sections into RQ1--RQ5.
- `[ADD]` Main result table now leads with clean/static/TCSLU fidelity.
- `[ADD]` CIFAR10-DVS Route 1 is the main event-data row: 82.3% teacher to
  83.9% LUT-only deployment on 1,000 validation samples.
- `[ADD]` User-authorized Route 3 is complete: 130 clean epochs, 81.2% best
  teacher, fixed 12-epoch LUT transfer, and 81.7% best LUT-only result.
- `[REWRITE]` The original fixed queue's missing-checkpoint stop is retained as
  history, followed by the later authorized from-scratch extension.
- `[PENDING]` ImageNet-100 remains an explicit pending row.
- `[PENDING]` DVS128 is labeled backbone-only; no LUT result is implied.
- `[KEEP]` E0/E1 occupancy and semantic controls form the main mechanism test.
- `[KEEP]` E5 is described as reliable backoff but partial compactness.
- `[KEEP]` E7 is the central compactness result with metadata-aware accounting.
- `[ADD]` A TCSLU component table separates static, temporal-only,
  channel-only, moment, and combined rows.
- `[REWRITE]` Original Spikformer is a mismatch boundary; QK-contract
  Spikformer is an architecture intervention.
- `[MOVE]` Grouped and native replacement results are compressed into one
  completeness-boundary table.
- `[CUT→SUPP]` Detailed all-affine, Q/K/V, MLP, patch, and classifier history
  moved to the appendix.

## Discussion and Conclusion

- `[REWRITE]` Discussion now centers on semantic addressability, dynamic
  contracts, explicit uncertainty, and system-measurement limits.
- `[ADD]` Explicitly separates local reconstruction from classification.
- `[REWRITE]` Conclusion answers what was discovered and what cannot be
  inferred; it no longer repeats every operator-audit number.
- `[PENDING]` No unfinished experiment is used as conclusion evidence.

## Traceability and Claim Boundaries

- E1/E5/E7 values: `results/EXPERIMENT_RESULTS_SUMMARY.md`.
- Current replacement and TCSLU rows:
  `paper/tables/data/tcslu_current_replacement_status.csv`.
- Metadata-aware storage:
  `paper/tables/data/lookup_cost_proxy.csv`.
- Native replacement audit:
  `results/qk_native_affine_bn_lif_transition_c100_20260619.md`.
- Grouped projection audit:
  `results/qk_grouped_projection_c100_all4_20260618.md`.

The manuscript does not claim measured SRAM, latency, energy, GPU-memory
savings, ImageNet success, reproduction of SpiLiFormer's published 86.7%,
an event-data result above 84.0%, universal spiking-neuron principles, or
complete production full-graph replacement.
