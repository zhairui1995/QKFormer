# QK-LUTFormer Paper Story Plan

Last updated: 2026-06-06

This note records the current paper direction, branch points, claim boundaries,
and experiment requirements for the QK-LUTFormer / QKFormer-LUT manuscript. It
is intentionally conservative: unverified claims are marked as hypotheses or
future work.

## Working Thesis

QK-LUTFormer should not be framed as a full LUT replacement for QKFormer yet.
The stronger and safer story is:

> Spiking Q-K attention exposes a compact discrete neural address space through
> binary Q/K/gate activity. We audit whether this address space is
> lookup-addressable, show where forward-only prototypes are insufficient, and
> test whether correct Q/K address alignment preserves local responses better
> than shuffled or non-address controls. Trainable LUT adapters remain a
> candidate method route, but frozen alignment is currently the cleanest proof
> of address semantics.

This makes the paper a diagnostic-to-adapter study, not a hardware deployment
paper and not a pure-spike Transformer replacement paper.

## Current Evidence Boundary

Supported by current E0-E2:

- A trained QKFormer checkpoint is non-silent on real CIFAR-10.
- Stage1/stage2 binary Q/K/gate addresses have useful occupancy and high hit
  rate.
- Address-wise prototypes modestly improve held-out reconstruction MSE over a
  global mean in E1.
- Stage1-only E2 replacement has a small positive validation signal.
- Blend 0.25 is more stable and lower-drift than full replacement in random
  calibration sweeps.
- Forward-only address specificity is weak because global-mean smoothing nearly
  matches address LUT in E2.

Not supported yet:

- Address LUT adapters improve accuracy beyond controls.
- Q/K addresses provide a hardware-efficient implementation.
- The method reduces energy, latency, spike rate, or switching activity.
- The method uses temporal coding, anchor selection, first-spike timing, or
  rank-order timing.
- A full LUT wrapper is justified.

Latest E3/E2 control update:

- The E3 trainable adapter path runs end-to-end.
- T=1 three-way conservative E3 showed `address_lut` was the only positive
  adapter versus `global_mean` and `token_channel_lut`.
- The stronger T=1 five-way E3 control weakened the trainable address-specific
  accuracy claim: `shuffled_address_lut` mean Acc@1 delta was +0.0967 versus
  `address_lut` +0.0600.
- The T=1 frozen E2 alignment test is now the cleanest positive evidence:
  frozen `address_lut` beats `shuffled_address_lut` and `global_mean` on
  Acc@1 delta, loss drift, KL, logit MSE, and local replacement MSE.

## Legacy Paper Migration

The earlier manuscript under the stopped SDR-LUTAttn / CCD-LUTAttn route should
be treated as historical negative evidence, not as the active paper method.

Reusable assets:

- The framing that Transformer attention should be studied as interaction
  preservation rather than operator-by-operator spike emulation.
- The negative diagnosis that pairwise addresses over DeiT softmax attention
  cannot identify the response because the softmax denominator depends on the
  whole row context.
- The reviewer-friendly habit of reporting localized replacement,
  progressive replacement, reconstruction error, and control ablations.
- The mathematical language for subspace/factorized LUTs, after rewriting it
  around QKFormer binary Q/K/gate addresses rather than anchor-order softmax
  pair addresses.

Must not migrate as current claims:

- CCD-LUTAttn as the proposed method.
- DeiT-Tiny full attention conversion as the active experimental setting.
- Anchor-based relative-order coding as an implemented QK-LUTFormer feature.
- Teacher-distilled all-head LUT conversion as a positive result.
- Any statement that full LUT conversion is viable after the old route.

How to use the old paper:

- Introduction: one short paragraph can motivate the pivot away from softmax
  pairwise response tables.
- Related work: keep ANN-to-SNN conversion and lookup-computation background,
  but reposition QK-LUTFormer under directly trained spiking Transformers and
  frozen-backbone adapters.
- Method: rewrite from pairwise softmax response lookup to module-local
  QKFormer `proj_lif` response adapters.
- Experiments: old results can appear only as motivation or appendix negative
  evidence if the target venue tolerates it; otherwise keep them in project
  docs and focus the paper on QKFormer E0-E3.

## Evidence Levels

- Level A: already supported by current real-data experiments.
- Level B: directly testable by E3 with existing code.
- Level C: low-cost extension requiring new but local code.
- Level D: future work needing new hardware/simulator integration or new
  training design.
- Level X: avoid; likely to be rejected as overclaim.

## Branch Points

### B1: Main Contribution

Options:

- B1-A, recommended now: Diagnostic-to-trainable residual LUT adapter.
  Evidence level: A for diagnostics, B for adapter claim.
  Risk: E3 may not beat controls. If E3 fails, the paper becomes a negative
  but rigorous feasibility study.

- B1-B: Forward-only LUT replacement.
  Evidence level: A but weak.
  Risk: E2 global-mean control nearly matches address LUT, so reviewers can
  argue the lookup address is not doing causal work.

- B1-C: Hardware LUT accelerator.
  Evidence level: D/X.
  Risk: no real hardware or energy measurement; only acceptable as simulated
  proxy/future work.

Current decision after E3 five-way and E2 frozen alignment: keep B1-A, but
split the contribution into two layers. The **address-space proof** should use
frozen prototype alignment. The **trainable adapter** should be presented as a
candidate route that currently needs better controls/design, not as the main
positive proof.

### B2: Temporal / Anchor Coding

Options:

- B2-A, recommended now: Mention temporal coding only as related work and future
  address extension.
  Evidence level: D.

- B2-B: Add time-step / first-spike / spike-count address features as E4.
  Evidence level: C after implementation.

- B2-C: Claim current method uses relative temporal coding.
  Evidence level: X.
  Risk: current addresses are binary Q/K/gate plus token/channel/head context;
  there is no verified first-spike or relative timing address.

### B3: Large LUT Compression

Options:

- B3-A, recommended near term: Add a small E4 factorized-address ablation using
  subtable composition across Q/K/gate, token, channel, and head bins.
  Evidence level: C.

- B3-B: Keep product-codebook / hash-embedding / tensor-factorized LUT as
  method motivation and future scalability.
  Evidence level: D.

- B3-C: Claim current LUT solves large sparse table scaling.
  Evidence level: X.

### B4: Sparsity Framing

Options:

- B4-A, recommended: Report sparsity as an address-space diagnostic with proxy
  metrics: active entries, hit rate, singleton fraction, occupancy entropy,
  samples per active entry, table footprint, lookup count, spike nonzero rate.
  Evidence level: A/C depending on metric.

- B4-B: Say sparse/event-driven structure is hardware-oriented and may reduce
  memory traffic under an event-driven implementation.
  Evidence level: D; phrase as motivation only.

- B4-C: Claim measured energy, latency, lower switching activity, or lower
  memory access frequency.
  Evidence level: X unless measured.

### B5: Hardware Evaluation

Options:

- B5-A, recommended for manuscript: Hardware-oriented simulation protocol:
  report operation proxies and table-memory proxies on CPU/GPU traces, with no
  real energy claim.
  Evidence level: C.

- B5-B: Apply for EBRAINS SpiNNaker/BrainScaleS or Intel INRC/Loihi access.
  Evidence level: D.

- B5-C: Claim deployment on neuromorphic hardware.
  Evidence level: X.

## Safe Paper Positioning

Title style:

- "QK-LUTFormer: Auditing and Adapting Spiking Q-K Attention with Binary
  Lookup Addresses"
- "From Spiking Q-K Attention to Trainable LUT Adapters: A Controlled Study of
  Discrete Neural Address Spaces"

Core contributions:

1. A non-invasive diagnostic framework for measuring binary Q/K/gate address
   coverage, occupancy, and conditional response variance in QKFormer.
2. Split-aware address prototype reconstruction and stage-wise replacement
   controls showing that forward-only LUT replacement has limited
   address-specific evidence.
3. A frozen prototype address-alignment test showing that correct Q/K address
   lookup preserves stage1 responses better than shuffled or global controls.
4. A frozen-backbone trainable residual LUT adapter study with strict controls:
   address LUT, global mean, shuffled address, and token/channel-only LUT,
   currently treated as a method route rather than the main proof.
5. A claim-faithful proxy protocol for table sparsity, lookup activity, and
   hardware-oriented cost, without claiming measured hardware gains.

Claims to avoid:

- "QK-LUTFormer is a full LUT implementation of QKFormer."
- "Our method reduces energy/latency on neuromorphic hardware."
- "The current model uses temporal coding or relative spike timing."
- "Sparse LUT occupancy proves lower firing rate or lower memory access."
- "Forward-only address prototypes are enough."

## Recommended Method Structure

1. Background: spiking Q-K attention as binary token/channel gating.
2. Address extraction: binary Q/K/gate, token/channel/head/stage context, and
   response targets.
3. E0 diagnostics: address coverage, occupancy, singleton fraction,
   conditional response variance.
4. E1 split-aware prototype reconstruction: calibration/evaluation split,
   address LUT vs global and coarse controls.
5. E2 controlled replacement: stage selection, blend, calibration stability,
   global and shuffled controls.
6. Frozen prototype alignment: same prototype bank, address vs shuffled vs
   global, local MSE and logit drift first, Acc@1 as downstream sanity.
7. E3 trainable residual LUT adapter: frozen backbone, shrinkage
   initialization, learnable alpha or fixed alpha, CE + KL + local MSE,
   matched controls.
8. Hardware-oriented proxy analysis: table footprint, active entries, lookup
   count, replaced operation proxy, no real energy claim.

## Recommended Experiment Matrix

Required for a credible positive paper:

| Experiment | Purpose | Minimum comparison | Claim enabled |
|---|---|---|---|
| E0 trained checkpoint | Address feasibility | per stage/block | binary Q/K/gate addresses are non-silent and occupied |
| E1 reconstruction | Address predictiveness | address vs global vs candidate/background | address prototypes carry modest response information |
| E2 replacement | Forward-only sanity | stage1, stage2, combined; blend; shuffled/global | forward-only replacement is limited |
| E2 frozen alignment | Address semantic proof | address vs shuffled vs global, same frozen prototype bank | Q/K addresses are meaningful LUT indices, if positive |
| E3 adapter | Method candidate | address vs global vs token/channel vs shuffled | trainable adapter exploits address structure, only if positive |
| E3 seeds | Robustness | at least 3 seeds | effect is stable, not single-run noise |
| E3 parameter budget | Fairness | matched or reported trainable params | gains are not just parameter count |
| Proxy cost | Hardware orientation | footprint, active entries, lookup count | method has lookup-friendly structure, not measured efficiency |

Optional E4:

| Experiment | Purpose | Claim enabled |
|---|---|---|
| Time feature address | Test temporal coding hypothesis | temporal address features help, if positive |
| Factorized LUT | Test scalable subtable design | factorization mitigates table sparsity, if positive |
| Calibration efficiency | Samples per active entry | data efficiency of address tables |

## Claim To Data Map

| Claim | Evidence required | Current status |
|---|---|---|
| Q/K/gate spikes define a compact discrete address space | E0 coverage, hit rate, occupancy, singleton fraction | supported for CIFAR-10 trained checkpoint |
| Address prototypes explain local responses better than global mean | E1 held-out MSE | supported, modest |
| Forward-only replacement is enough | E2 downstream + controls | not supported |
| Frozen address alignment matters | same prototype bank, address beats shuffled/global on local MSE and drift | supported once on T=1 stage1 |
| Trainable address LUT adapter uses address semantics | conservative E3 address beats global, token/channel, and shuffled under same budget | not supported at current setting; shuffled trainable control is too strong |
| LUT sparsity can be exploited | active entries, entropy, hit rate, sparse storage ratio | partially pending |
| Hardware efficiency | real hardware or simulator energy model | not available |
| Temporal coding improves LUT address | first-spike/time-step/spike-count ablation | not implemented |

## Related Work Search Directions

Spiking Transformers:

- QKFormer, Spikformer, Spiking Self-Attention, spike-driven Transformer,
  direct-training SNN Transformer.

Temporal coding:

- time-to-first-spike, TTFS, rank-order coding, latency coding, N-of-M coding,
  sparse temporal coding, first-spike SNN.

Lookup / table compression:

- feature hashing, hash embeddings, product quantization, compositional
  codebooks, Cartesian product codebooks, multi-table lookup, tensor train,
  CP/Tucker factorization, LUTNet, LogicNets, FPGA LUT neural inference.

Hardware/sim:

- Lava, Loihi 2, Intel Neuromorphic Research Cloud, SpiNNaker, BrainScaleS,
  EBRAINS NMC, SpikingJelly, snnTorch, PyNN, event-driven SNN simulator.

## Draft Introduction Framing

Spiking Transformers promise event-driven computation, but their internal
discrete structure is rarely tested as an addressable computational object.
QKFormer is an especially useful testbed because its Q-K attention replaces
softmax-style pairwise attention with binary spike-form token/channel gating.
This raises a concrete question: do the binary Q/K/gate patterns learned by a
trained spiking Transformer form a compact, reusable lookup address space, or
are they merely incidental activations that fail under controlled replacement?
We answer this question through a staged audit. First, we measure address
occupancy and conditional response variance in a trained QKFormer. Second, we
calibrate split-aware response prototypes and test whether address-conditioned
lookup improves held-out reconstruction. Third, we perform controlled
stage-wise replacement and show the limits of forward-only prototypes. Fourth,
we freeze calibrated prototypes and show that correct Q/K address alignment
preserves stage1 responses better than shuffled or global controls. Finally, we
study trainable residual LUT adapters as a candidate route and show why
matched shuffled controls are necessary. This framing separates what binary
spike addresses already support from what remains a future adapter or hardware
deployment problem.

## Skeptical Reviewer Risks

- "This is just smoothing, not address semantics." Mitigation: the frozen E2
  alignment test must show address LUT beating global mean and shuffled address
  under the same prototype bank; trainable E3 alone is not enough.
- "The gains are tiny." Mitigation: position E0/E1/E2 as diagnostics, use
  local MSE/KL/logit drift as mechanistic evidence, and make E3 the method only
  if it gives stable gains or better accuracy/drift trade-offs against
  shuffled controls.
- "CIFAR-10 is too small." Mitigation: add CIFAR-100 if feasible; otherwise
  explicitly present CIFAR-10 as a feasibility study.
- "No hardware result." Mitigation: call it hardware-oriented proxy analysis,
  not hardware deployment; report table/lookup metrics only.
- "Temporal coding is overclaimed." Mitigation: keep temporal coding in related
  work/future work unless E4 is implemented.
- "Large LUT scaling is not solved." Mitigation: present factorized/subtable
  LUT as optional E4/future scalability, not as a current result.

## Source Notes

- QKFormer proposes spike-form Q-K attention over binary vectors, linear
  token/channel complexity, sparse additions, and mask operations.
- TTFS and rank-order coding literature supports temporal-code motivation, but
  not the current binary-address implementation.
- Product quantization, hash embeddings, and tensor-train neural compression
  support the idea of factorized/compositional tables, but require experiments
  before becoming a claim.
- Lava can prototype on CPU/GPU and target neuromorphic backends such as Loihi;
  Intel INRC describes remote Loihi access for members; EBRAINS describes free
  test access for accepted SpiNNaker/BrainScaleS use. These are access paths,
  not current project evidence.
