# QK-LUTFormer Innovation Story

Last updated: 2026-06-06

This document packages QK-LUTFormer into a coherent research story with more
than three LUT-centered innovation points. It reflects the current evidence:
E0/E1 are `CONDITIONAL GO`, E2 shows that forward-only address prototypes are
limited, E3 trainable five-way controls show that trainable tables can blur
address-specific claims, and the latest frozen E2 alignment test provides the
cleanest current support for Q/K binary addresses as a LUT indexing signal.

## Executive Thesis

The paper should not claim that QKFormer can already be converted into a full
lookup-table Transformer. The more defensible and more interesting story is:

> Softmax attention resists pairwise LUT conversion because pair addresses do
> not encode row-normalization context. QKFormer offers a different substrate:
> its spike-form Q-K attention already exposes binary Q/K/gate activity. We
> study whether these internal spike patterns form a reusable discrete neural
> address space for lookup-conditioned local adaptation. The answer is nuanced:
> the addresses are occupied and modestly predictive, naive prototypes are
> insufficient as a full wrapper, trainable residual tables require strict
> controls, and frozen address-alignment tests show that correct Q/K address
> alignment is more informative than shuffled alignment.

This turns the paper into a staged addressability study with a conservative
LUT-adapter route, rather than a fragile positive-only wrapper claim. The main
positive method evidence should be framed around **frozen address alignment**;
the trainable E3 result should be used as a cautionary control showing why
parameter-matched shuffled tables are necessary.

## Recent Literature Signals

The latest literature strengthens, but also constrains, our positioning:

- **QKFormer / spiking Transformers**: QKFormer uses spike-form Q-K attention
  with binary vectors for token/channel attention and linear complexity. 2025
  spiking Transformer papers continue to move toward addition-only, saccadic,
  quantized, and theory-guided spike attention rather than softmax emulation.
  Sources:
  - https://proceedings.neurips.cc/paper_files/paper/2024/hash/179f5dcdeedc149443ebd3ba70811dbd-Abstract-Conference.html
  - https://arxiv.org/abs/2503.00226
  - https://arxiv.org/abs/2502.12677
  - https://arxiv.org/abs/2501.13492
  - https://arxiv.org/abs/2604.15769

- **LUT-native neural networks**: 2025-2026 work such as LL-ViT, SparseLUT,
  BitLogic, and HGQ-LUT emphasizes that good LUT networks are trained or
  connectivity-optimized as LUT-native structures; they are not merely
  arithmetic operations replaced by lookup after the fact.
  Sources:
  - https://arxiv.org/abs/2511.00812
  - https://arxiv.org/abs/2503.12829
  - https://arxiv.org/abs/2601.09773
  - https://arxiv.org/abs/2602.07400
  - https://arxiv.org/abs/2604.22293

- **Lookup experts and adapters**: MoLE turns trained experts into lookup
  tables before inference; adapter/LoRA literature supports frozen-backbone
  small-module adaptation, but not address semantics by itself.
  Sources:
  - https://arxiv.org/abs/2503.15798
  - https://proceedings.mlr.press/v97/houlsby19a.html
  - https://arxiv.org/abs/2106.09685

- **Temporal coding**: TTFS, rank-order, and temporal sequence learning provide
  future address features, but current QK-LUTFormer does not yet use them.
  Sources:
  - https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.971937/full
  - https://proceedings.neurips.cc/paper/2020/hash/8bdb5058376143fa358981954e7626b8-Abstract.html

- **Hardware pathway**: Lava and EBRAINS provide simulation/hardware-access
  paths, while FPGA LUT papers justify table-size and connectivity proxies.
  These do not justify energy claims without measurement.
  Sources:
  - https://lava-nc.org/
  - https://ebrains.eu/data-tools-services/computing-infrastructure/neuromorphic-computing
  - https://wiki.ebrains.eu/bin/view/Collabs/neuromorphic/Getting%20access/

## Innovation 1: Spike-Addressability Audit

### Core Idea

Treat binary Q/K/gate activity inside QKFormer as a discrete neural address
space, then audit whether it is useful before building any LUT method.

This is the foundation contribution. It is stronger than saying "we made a
LUT," because it gives reviewers a falsifiable measurement protocol:

- address coverage,
- hit rate,
- singleton fraction,
- bucket occupancy entropy,
- conditional response variance,
- calibration samples per active entry,
- stage/block/module-wise address usefulness.

### Why It Matters

The old DeiT/CCS route failed because pairwise softmax response LUTs missed
row context. The audit explains why QKFormer is a better substrate: QKFormer
already computes attention through spike-form Q/K/gate patterns, so its
internal binary variables are not artificial quantization bins added after
training.

### Evidence Status

Supported by current E0/E1 and latest frozen E2 alignment:

- trained checkpoint is non-silent,
- stage1/stage2 addresses are well occupied,
- E1 address prototypes modestly beat global mean reconstruction,
- T=1 frozen address LUT beats shuffled address and global mean on local
  replacement MSE, logit drift, loss, and Acc@1 delta,
- effect is real but still limited to CIFAR-10 and stage1 evidence.

### Safe Claim

"We introduce a staged audit for lookup-addressability in spiking Q-K attention
and show that trained QKFormer early-stage addresses are occupied, modestly
predictive, and more useful than shuffled addresses in a frozen prototype
alignment test."

### Unsafe Claim

"The audit proves QKFormer can be replaced by a LUT" or "the addresses are
already hardware efficient."

## Innovation 2: Negative-Controlled Forward LUT Replacement

### Core Idea

Make the negative result part of the contribution: forward-only address
prototypes are not enough, and global smoothing nearly matches them. This
control prevents us from overclaiming address semantics.

### Why It Matters

Many LUT papers can be attacked with "you are just smoothing/averaging." E2
turns that criticism into an experimental gate:

- compare address LUT to global mean,
- compare address LUT to shuffled address LUT,
- compare blend strengths,
- compare stage1/stage2/combined targets,
- report logit KL and local MSE, not only accuracy.

### Evidence Status

Supported by current E2:

- stage1-only has small positive full-validation signal,
- stage2 and combined are negative,
- blend 0.25 is more stable/lower drift,
- global mean nearly matches address LUT and has better loss in one control
  sweep,
- E3 trainable shuffled-address control can match or exceed address LUT in
  mean Acc@1 delta, showing that trainable-table gains are not automatically
  address-semantic.

### Safe Claim

"Forward-only prototype lookup is measurable but not sufficient as a wrapper:
its downstream benefit is small, and trainable variants need frozen or
shuffled controls to separate address semantics from smoothing and table
regularization."

### Unsafe Claim

"Address LUT replacement improves QKFormer" as a general statement.

## Innovation 3: Drift-Constrained Residual Address LUT Adapter

### Core Idea

Use a frozen QKFormer backbone and train only a small residual LUT adapter
indexed by Q/K-derived addresses:

```text
y = y_original + alpha * (LUT[address] - y_original)
```

The first E3 pilot showed that unconstrained learnable alpha drifts too much:
address LUT was least damaging, but all modes degraded accuracy. That suggests
the right method is not "train the LUT harder"; it is:

- small fixed or strongly regularized alpha,
- stronger KL-to-baseline,
- stronger local-MSE drift constraint,
- shrinkage initialization,
- same-budget controls.

### Why It Matters

Recent LUT-native papers argue that LUT modules need training-aware design.
Our version is not a generic LUT neuron; it is a **spike-address-conditioned
adapter**. The adapter is only publishable as an address-specific method if it
beats controls.

### Evidence Status

Mixed:

- E3 code path works.
- T=1 three-way conservative E3 made `address_lut` the only positive adapter
  versus `global_mean` and `token_channel_lut`.
- The stronger five-way E3 control weakened that claim because
  `shuffled_address_lut` had mean Acc@1 delta +0.0967 versus `address_lut`
  +0.0600.
- Therefore trainable E3 is not the current proof of address semantics.
- E3 remains useful as a method candidate, but only after a frozen-alignment
  mechanism or stricter parameter/control design prevents shuffled tables from
  adapting around broken address maps.

### Required Experiments

Future E3 should not chase seed-level gains first. It should revise the method
so address semantics are identifiable:

- include `shuffled_address_lut` as a required matched-parameter control;
- report whether trainable shuffled tables can recover from broken alignment;
- consider freezing the prototype table and training only a small global or
  module-level blend/gate;
- compare against the frozen E2 alignment result as the address-semantic
  anchor.

### Safe Claim If Positive

"A drift-constrained residual LUT adapter can exploit Q/K-derived addresses
better than non-address and weakened-address controls under a frozen-backbone
budget."

### Safe Claim If Negative

"Even after trainable residual adaptation, the current Q/K address definition
does not provide enough address-specific benefit beyond smoothing controls."

### Unsafe Claim

"E3 already improves accuracy" or "the trained LUT adapter is the final method"
before conservative controls pass.

## Innovation 3b: Frozen Prototype Address-Alignment Test

### Core Idea

Freeze the calibrated LUT prototypes and compare correct Q/K address lookup
against shuffled-address lookup under the same calibration split and full
validation evaluation. Unlike trainable E3, the shuffled control cannot
relearn around a broken address/prototype mapping.

### Why It Matters

This is currently the most reviewer-friendly test of the core methodology. It
answers the sharp question:

> Are Q/K binary spike addresses meaningful indices into local response
> prototypes, or are LUT gains merely smoothing/regularization?

### Evidence Status

Supported by T=1 E2 frozen alignment:

- Result: `results/qkformer_lut_e2_alignment_test_20260606_182021`.
- Address LUT: Acc@1 94.9000 -> 94.9400, delta +0.0400; local MSE 0.055959.
- Shuffled-address LUT: Acc@1 94.9000 -> 94.8400, delta -0.0600; local MSE
  0.063405.
- Global mean: Acc@1 94.9000 -> 94.7900, delta -0.1100; local MSE 0.059798.
- Address LUT also has lower loss drift, KL, and logit MSE than both controls.

### Safe Claim

"When LUT prototypes are frozen, correct Q/K spike-address alignment preserves
the QKFormer stage1 response better than shuffled or global controls."

### Unsafe Claim

"Frozen prototype replacement is a finished deployable LUT layer" or "the
observed Acc@1 change proves a general accuracy improvement."

## Innovation 4: Factorized / Connectivity-Optimized Address Tables

### Core Idea

A full Q/K/gate/token/channel/head address table is sparse and may overfit.
Use multiple smaller subtables rather than one large monolithic table:

- Q/K bit subtable,
- token-bin subtable,
- channel-bin subtable,
- gate/population subtable,
- head/stage subtable,
- optional temporal subtable later.

Combine outputs additively or through learned low-rank/gated composition.

### Why It Matters

Latest LUT DNN work repeatedly identifies LUT size, fan-in, and connectivity
as first-order issues. SparseLUT/TCAD-style connectivity optimization and
BitLogic/HGQ-LUT-style trainable LUT workflows support the idea that table
structure is a method design axis, not a bookkeeping detail.

### Evidence Status

Not implemented yet. This is the strongest E4 candidate if E3 conservative
still fails or is weak.

### Low-Cost E4 Design

Add adapter modes:

- `factorized_sum_lut`: sum of QK, token, channel, gate subtables.
- `factorized_gated_lut`: learned scalar gates per subtable.
- `hash_address_lut`: hashed address with collision control.
- `matched_param_address_lut`: address LUT with parameter count matched to
  factorized mode.

### Safe Claim If Positive

"Factorized address tables improve the accuracy/drift/parameter trade-off over
monolithic address LUTs under sparse occupancy."

### Unsafe Claim

"We solve LUT explosion" unless we report table memory, active entries, and
accuracy across address widths.

## Innovation 5: Temporal-Population Address Extension

### Core Idea

Do not claim temporal coding yet. Instead, propose and optionally test whether
time-step features improve LUT addresses:

- first active timestep of Q/K/gate,
- spike count over T,
- time-binned population activity,
- relative first-spike order between Q and K features.

### Why It Matters

Temporal coding literature gives a real theoretical base, and 2025 spiking ViT
work emphasizes spatial-temporal spike interactions. But the current address
is binary activation/context, so timing must be an ablation, not a claim.

### Evidence Status

Not implemented. Future/E4 only.

### Safe Claim If Implemented

"Adding simple temporal-population address fields tests whether QKFormer's
time dimension provides lookup-relevant information beyond binary activation."

### Unsafe Claim

"QK-LUTFormer uses rank-order or TTFS coding" before implementing timing
features.

## Innovation 6: Hardware-Oriented LUT Proxy Protocol

### Core Idea

Report reproducible proxies that align with LUT hardware concerns without
claiming measured deployment:

- address bit width,
- dense table entries,
- active entries,
- sparse storage ratio,
- lookup count per image,
- replaced response values,
- spike nonzero rate,
- average samples per active bucket,
- estimated original MAC/add count avoided by the diagnostic replacement.

### Why It Matters

LUT/FPGA papers are judged by fan-in, table size, connectivity, and bit-exact
workflow. Neuromorphic claims require real hardware or a simulator, which we
do not yet have. A proxy protocol is honest and still useful.

### Evidence Status

Partially implemented in E0/E1/E2 metrics; needs a result parser and table.

### Safe Claim

"We provide hardware-oriented table and lookup proxies to characterize whether
the address space is plausible for future LUT-oriented implementations."

### Unsafe Claim

"QK-LUTFormer reduces energy/latency" without measurement.

## Complete Paper Story

### Act 1: Why Softmax LUT Failed

The old SDR/CCS line is negative evidence: softmax attention responses depend
on row context, so pairwise address prototypes collapse under full
replacement. This is the reason to stop wrapper engineering, not an
embarrassment to hide.

### Act 2: Why QKFormer Is the Right Substrate

QKFormer's spike-form Q-K attention already exposes binary Q/K/gate variables.
Instead of inventing anchor-order codes over continuous DeiT features, we audit
learned spike addresses inside a trained spiking Transformer.

### Act 3: What the Audit Shows

E0/E1 say the address space is real but imperfect:

- occupied enough to study,
- modestly predictive,
- high enough variance that naive replacement is risky.

This creates a careful `CONDITIONAL GO`, not a full method claim.

### Act 4: What the Controls Show

E2 says forward-only prototypes are not address-specific enough:

- stage1-only small signal,
- stage2 negative,
- global smoothing nearly matches address LUT.

This prevents a weak paper from pretending smoothing is address semantics.

### Act 5: The Proposed Method

The method should now be described in two layers:

1. **Address-alignment proof**: frozen prototype lookup demonstrates that Q/K
   address alignment matters.
2. **Adapter route**: drift-constrained residual LUT adaptation is a candidate
   way to turn address alignment into trainable modules, but it needs stricter
   controls because trainable shuffled tables can adapt.

The trainable adapter route still follows:

- freeze QKFormer,
- initialize LUT from address prototypes,
- constrain alpha/local drift,
- train tiny address-indexed tables,
- require controls to pass.

If conservative E3 succeeds under shuffled controls, it can become the main
positive contribution. If it does not, the paper can still use frozen
alignment as the main address-space result and move trainable adapters to an
open method-design problem.

### Act 6: Hardware Orientation Without Hardware Overclaim

We report table/lookup proxies and connect them to LUT-native literature.
Actual FPGA/neuromorphic deployment is future work unless we synthesize or
port to Lava/EBRAINS.

## Recommended Paper Title Options

Safe and clear:

- "QK-LUTFormer: Auditing Binary Q-K Spike Addresses for Lookup-Based
  Adaptation"
- "Are Spiking Q-K Patterns Lookup Addresses? A Controlled Study with
  QKFormer"
- "From Softmax LUT Failure to Spike-Addressed LUT Adapters in Spiking
  Transformers"

More ambitious, only if E3/E4 positive:

- "QK-LUTFormer: Drift-Constrained Lookup Adapters for Spiking Q-K Attention"
- "Spike-Addressed Lookup Adapters for Spiking Vision Transformers"

Avoid:

- "Hardware-Efficient QK-LUTFormer"
- "Pure LUT Transformer"
- "Temporal LUTFormer"

## Claim Ladder

### Already Supported

- Trained QKFormer has non-silent Q/K/gate activity on CIFAR-10.
- Stage1/stage2 addresses have strong occupancy.
- Address prototypes modestly improve reconstruction MSE.
- Forward-only downstream replacement is limited and weakly address-specific.
- T=1 frozen address alignment beats shuffled/global controls in the stage1
  replacement test.
- E3 trainable adapters work technically, but five-way controls show the
  current E3 accuracy signal is not cleanly address-specific.

### Testable Now

- Repeat frozen alignment on another checkpoint or stage1 target.
- Optimize the alignment calibration path for 512-batch/full-train evidence.
- Revise E3 so trainable tables cannot hide broken address alignment.
- Proxy-cost parsing can quantify table sparsity and lookup activity.

### E4 / Future

- Factorized address LUTs.
- Temporal-population address fields.
- CIFAR-100 / more stage1 targets.
- Lava/EBRAINS/FPGA hardware experiments.

### Avoid

- Accuracy improvement claim from E3 pilot.
- Energy/latency claim.
- Full wrapper claim.
- Temporal coding claim.

## Experiment Gate For Story Choice

### Gate A: Frozen Alignment Positive

Condition:

- frozen `address_lut` beats `shuffled_address_lut` and `global_mean` on local
  MSE, KL/logit drift, and non-degrading Acc@1 under the same calibration bank.

Story:

- "Q/K binary spike patterns are meaningful LUT addresses; trainable adapters
  are a method route, not the proof itself."

Current status: passed once on T=1 CIFAR-10 stage1 with 128 calibration batches
and full validation. Repeat before broadening the claim.

### Gate B: Trainable E3 Positive Under Shuffled Control

Condition:

- trainable `address_lut` beats `global_mean`, `token_channel_lut`, and
  `shuffled_address_lut` under the same frozen-backbone budget.

Story:

- "Drift-constrained spike-addressed LUT adapters convert address alignment
  into a trainable module."

Current status: not passed. Five-way E3 is `NO-GO` for address-specific
accuracy at the current setting because shuffled address has higher mean
Acc@1 delta.

### Gate C: Frozen Alignment Fails To Repeat

Condition:

- frozen address LUT no longer beats shuffled/global on another checkpoint,
  stage, or larger dataset.

Story:

- "Q/K addresses carry weak but real local structure; monolithic residual
  adapters are too blunt. Move to factorized address tables."
- "Current binary Q/K address definition is not sufficient for address-specific
  LUT adaptation. The paper becomes a negative feasibility audit unless a new
  address design is introduced."

Next method:

- temporal-population address, factorized address, or stop as negative result.

## Reviewer One-Liner

If a reviewer asks "what is the real contribution?", the answer should be:

> We do not merely replace attention with a table. We introduce a controlled
> addressability protocol for spiking Q-K attention, show that naive prototypes
> fail important smoothing controls, and show that frozen Q/K address-aligned
> prototypes preserve stage1 responses better than shuffled or global controls.
> Trainable LUT adapters are the next method route, but not the current proof
> of address semantics.
