# QK-LUT-LIF Architecture Acceptance Pipeline

Date: 2026-07-01

Owner role: `ccf-pipeline-orchestrator`.

This addendum adapts the earlier CCFA acceptance pipeline from a diagnostic
LUT spiking-neuron boundary paper to the current third-paper goal: a proposed
spiking Transformer architecture derived from the second paper's QK-LUT
principles.

## Current Stage

Stage: R3/R4 evidence acquisition for an architecture pivot.

The paper is not yet allowed to claim a successful new architecture or
near-SOTA result. It currently has:

- three completed `NO-GO` frozen/post-hoc architecture probes;
- attention-scope native QK-LUT-LIF smoke pass;
- LL-ViT-inspired MLP-scope native QK-LUT-LIF smoke pass;
- two fixed 30-epoch pilots running on server-lbz.

The target identity under test is:

> QK-LUT-LIF Transformer: a QK-LUT-inspired spiking Transformer family that
> inserts lookup-parameterized LIF transition modules into trainable network
> blocks rather than fitting a frozen backbone after training.

## Evidence Gate

No manuscript section should state the architecture as successful until at
least one native pilot passes its registered gate.

| Route | Run spec | Current status | Pass gate |
|---|---|---|---|
| Attention-scope native QK-LUT-LIF | `run_specs/native_qklut_lif_attention_c100_t4_seed42_20260701.yaml` | pilot running | native best Acc@1 no worse than 1.00 pp below matched QKFormer baseline |
| LL-ViT-inspired MLP-scope native QK-LUT-LIF | `run_specs/native_qklut_lif_mlp_c100_t4_seed42_20260701.yaml` | pilot running | native best Acc@1 no worse than 1.00 pp below matched QKFormer baseline |

If both pilots fail, the architecture route remains unsupported and must move
to a failure-analysis or new-design gate, not to a full run.

If one pilot passes, the next action is a full fixed-budget run for that route
and a manuscript outline update that labels the result as a candidate
architecture, not yet a broad/SOTA result.

If both pilots pass, compare the smaller gap, table footprint, and target
interpretability, then select one primary route for the full run. Do not run
both full routes unless a new run spec justifies the cost and claim.

## Claim-Evidence Matrix

| Claim candidate | Required evidence | Current status |
|---|---|---|
| Native LUT-LIF can be trained as a module inside QKFormer. | At least one smoke pass and one pilot pass with matched baseline. | smoke passed; pilot pending |
| QK-LUT principles motivate insertion site. | Target names, table footprint, and a mechanistic link to attention Q/K or MLP channel mixer. | partially prepared |
| Architecture improves or preserves QKFormer-level accuracy. | Full run at registered budget; native best Acc@1 at least matched baseline. | missing |
| Near-SOTA spiking Transformer result. | Full comparison against current strong baselines on accepted datasets, preferably repeated or justified. | missing |
| Cross-architecture transfer to Spikformer/SpikeFormer/LL-ViT-like settings. | Separate preregistered runs and matched baselines. | missing |

## Experiment Queue

1. Finish attention-scope native 30-epoch pilot.
2. Finish MLP-scope native 30-epoch pilot.
3. Record both pilot results with exact `comparison.json`, `summary.csv`, and
   launcher provenance.
4. If a pilot passes, launch only the winning route's full 400-epoch run.
5. If no pilot passes, write a failure analysis and design a new module rather
   than sweeping learning rate, seed, bit width, or checkpoint.
6. After a full positive QKFormer result, design a cross-backbone route:
   Spikformer/SpikeFormer support first if runnable in this repo; LL-ViT is a
   literature/insertion-site baseline unless an official implementation is
   located and licensed for use.

## Writing Gate

Before rewriting the manuscript as a new architecture paper, create or update:

- a method section that names the architecture and separates it from the
  earlier diagnostic CNL-LUT-LIF boundary study;
- a result table with baseline QKFormer, attention-scope native, MLP-scope
  native, and post-hoc `NO-GO` controls;
- a claim ledger where every architecture claim maps to smoke, pilot, full,
  or cross-architecture evidence;
- a limitations paragraph preserving failed post-hoc and RL-selector probes.

## Disallowed Until Evidence Exists

- near-SOTA accuracy;
- superiority over QKFormer, Spikformer, SpikeFormer, or LL-ViT;
- broad transfer across architectures or datasets;
- hardware speed, latency, energy, SRAM, or area;
- treating smoke results as accuracy evidence;
- presenting LL-ViT-inspired MLP insertion as an LL-ViT reproduction.

## Next Concrete Action

Monitor the two running pilots and record whichever completes first. Do not
start a full run until a pilot result passes its preregistered gate.
