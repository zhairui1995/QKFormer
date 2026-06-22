# Paper TBD And Remaining Experiment Audit

Status date: 2026-06-22

## Raw Count

- `paper/sections/4_cross_arch_tables.tex` contains 47 uses of `\tbd`.
- Two uses occur in explanatory captions, leaving 45 unresolved table cells.
- The 45 cells do not correspond to 45 experiments. They combine pending
  experiment rows, uncomputed analytical fields, and missing literature or
  model metadata.

## Experiment-Bearing TBD Groups

### 1. CIFAR10-DVS event-data package

Current status: completed below the 84.0% deployment gate.

- Route 1 is the main event-data result: 82.3% clean teacher and 83.9%
  LUT-only Acc@1 on the complete 1,000-sample validation split.
- The original fixed queue did not launch Route 3 because no official
  CIFAR10-DVS checkpoint was released.
- A later user-authorized extension trained SpiLiFormer-2-256 from scratch for
  130 effective epochs, reaching 81.2%, then completed the fixed 12-epoch LUT
  transfer and reached 81.7%.
- Both deployment paths use per-time/per-channel current normalization and
  bypass the replaced original projection.

No CIFAR10-DVS experiment remains TBD for the current route package. Route 3
does not reproduce the published 86.7% clean result and should remain a bounded
cross-architecture implementation row.

### 2. Unmodified second-architecture transfer package

Current table target: SpikingResformer.

Required stages:

1. Reproduce or load a fixed clean baseline.
2. Run static and calibrated current replacement.
3. Run aligned, shuffled, token/channel, global, and random controls.

This experiment is required only if the paper keeps a broad
`cross-architecture generalization` claim. Otherwise, remove the pending
SpikingResformer rows and use the narrower phrase
`QK-contract-conditioned transfer evidence`.

### 3. Original Spikformer matched controls

The mismatch-probe row has a measured aligned result, but shuffled,
token/channel, global, and random cells remain open.

Recommendation: do not run these by default. The unmodified Spikformer result
is already framed as a negative compatibility boundary. Either remove the open
control cells or run one fixed five-way control package only if a reviewer
requires a complete mismatch comparison.

### 4. QK-contract Spikformer control completion

Aligned, shuffled, and token/channel are complete. Global and same-size random
controls remain open.

This is one small matched-control completion package. It would improve table
symmetry but would not resolve the existing fact that token/channel
slightly exceeds aligned lookup.

### 5. Original Spikformer robustness matrix

Four rows remain open:

- calibration-subset robustness;
- support-threshold robustness;
- noise/brightness corruption;
- multi-current replacement.

These are four separate experiment groups. They are not necessary while
unmodified Spikformer remains a mismatch probe. Remove the rows unless the
paper promotes unmodified Spikformer into a positive transfer claim.

## Non-Experiment TBD Groups

These should be completed by analysis or documentation rather than GPU runs:

- QKFormer calibration-subset static aggregate: derive only if the original
  static rows are traceable; otherwise use `N/A`.
- Spikformer address bits and lookup/MAC ratio.
- Grouped-projection metadata-inclusive KiB.
- Published-model metadata: QKFormer time steps, SpikingResformer parameter
  count, and Meta-SpikeFormer time steps.
- Parameter counts for the reported TCSLU-QKFormer and TCSLU-Spikformer units.

## ImageNet-100

The main manuscript retains one pending ImageNet-100 row. Any future run is
locked to the fixed seed-42 ImageNet-100 manifest defined in `AGENTS.md`; no
ImageNet-mini or alternative subset is authorized. Before promoting a result,
record:

- exact dataset construction and class/sample counts;
- architecture and checkpoint;
- time steps;
- clean baseline;
- replacement target;
- aligned, shuffled, token/channel, global, and random controls;
- calibration and evaluation budgets;
- near-lossless success gate.

Treat it as a new cross-dataset scale experiment, not as a retrospective fill
for unrelated cross-architecture cells.

## Practical Count

If every pending row is retained, the paper implies seven experiment packages:

1. SpikingResformer main replacement.
2. SpikingResformer semantic controls.
3. Original Spikformer semantic controls.
4. QK-contract Spikformer global/random completion.
5. Original Spikformer calibration-subset robustness.
6. Original Spikformer support/corruption robustness.
7. Original Spikformer multi-current replacement.

This is not the recommended path.

The remaining minimum high-value path is:

1. Preregister and complete the locked ImageNet-100 track if scale
   generalization is central.
2. Either complete one unmodified second-architecture matched-control package,
   or narrow the cross-architecture claim and delete its pending rows.

All other open cells should be filled analytically, changed to `N/A`, moved to
supplementary material, or removed before submission.
