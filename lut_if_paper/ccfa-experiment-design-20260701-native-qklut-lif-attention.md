# CCFA Experiment Design: Native QK-LUT-LIF Attention Training

Date: 2026-07-01

Mode: experiment design and execution queue.

## Venue And Assumptions

- Target: AAAI-style CCF-A main-track evidence package.
- Paper identity under test: a QK-LUT-inspired spiking Transformer
  architecture, not another post-hoc frozen-backbone lookup replacement.
- Completed post-hoc routes are all `NO-GO`: RL selector, dense trainable
  CNL-LUT-LIF, and structured residual CNL-LUT-LIF.
- The next credible architecture route must train the model with the
  QK-LUT-LIF module present from initialization.

## Claim-Evidence Matrix

| Claim | Reviewer question | Evidence needed | Dataset/benchmark | Baselines | Metrics | Result placeholder | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native attention-scoped QK-LUT-LIF can train end-to-end. | Does the new module participate in normal optimization rather than post-hoc fitting? | Matched QKFormer and native QK-LUT-LIF smoke/pilot runs from initialization. | CIFAR-100, QKFormer, `T=4`, seed 42. | Original QKFormer trained by the same runner and budget. | train loss, Acc@1/Acc@5, module count, table entries, non-finite checks. | TBD | planned |
| Native insertion is a stronger route than frozen-backbone insertion. | Does the post-hoc failure disappear when upstream layers can adapt? | Compare native pilot against the three completed post-hoc `NO-GO` reports. | Same as above. | RL selector, dense trainable CNL, structured residual CNL. | pilot Acc@1 gap, stability, failure mode. | TBD | planned |
| QK-LUT principles motivate the architecture. | Is this just adding arbitrary parameters? | Target only attention LIF nodes first, report selected target names and table footprint. | Same as above. | Baseline QKFormer. | target coverage, table entries, accuracy. | TBD | planned |

## Registered Experiment

Run spec:
`run_specs/native_qklut_lif_attention_c100_t4_seed42_20260701.yaml`.

Runner:
`scripts/server/run_native_qklut_lif_cifar100.sh`.

Initial target scope:

- attention-scoped LIF nodes only: names under `.tssa.` or `.ssa.`;
- 6-bit state and 8-bit input addressing;
- static current range `[-8, 8]`, static state range `[0, 2]`;
- entire network trains from initialization.

## Gates

Smoke gate:

- matched baseline and native rows complete one fixed short run;
- native mode selects at least one attention LIF target;
- training and validation finish without non-finite loss.

Pilot gate:

- both rows use identical seed, augmentations, epoch budget, and validation;
- native best Acc@1 is no worse than 1.00 pp below the matched QKFormer
  baseline;
- no learning-rate, bit-width, seed, target-scope, or checkpoint change is made
  after seeing smoke unless a new run spec is written.

Full gate:

- native best Acc@1 is at least as high as the matched QKFormer baseline under
  the full registered budget;
- near-SOTA language remains forbidden until the result is compared against
  current strong Spiking Transformer baselines and preferably repeated.

## Execution Priority

| Priority | Experiment | Claim defended | Cost | Dependency | Stop condition |
| --- | --- | --- | --- | --- | --- |
| P0 | smoke baseline + native attention QK-LUT-LIF | feasibility | low | code compiles on server | stop on any runtime/non-finite failure |
| P1 | 30-epoch pilot baseline + native | architecture viability | medium | P0 pass | stop after fixed pilot, no hidden sweep |
| P2 | full 400-epoch baseline + native | main architecture claim | high | P1 pass and user-visible decision | stop after fixed full run |

## No-Fabrication Status

No result is generated in this memo. The current implementation and run spec
only create a route to test whether a real native QK-LUT-LIF architecture can
train. They do not support near-SOTA, LL-ViT, Spikformer, hardware, latency,
energy, SRAM, or broad-transfer claims.
