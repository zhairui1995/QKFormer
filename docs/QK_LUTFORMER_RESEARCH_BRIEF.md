# QK-LUTFormer Research Brief

Last updated: 2026-06-06

Purpose: turn the current QK-LUTFormer branch into a credible paper story with
strict claim boundaries. This brief complements
`docs/QK_LUTFORMER_PAPER_STORY_PLAN.md`.

## One-Sentence Paper Direction

QK-LUTFormer is a controlled study of whether the binary Q/K/gate activity
inside a trained spiking Q-K Transformer forms a useful discrete neural address
space for lookup-based local adaptation.

The paper should be framed as:

- an address-space audit,
- a negative result for naive forward-only prototypes,
- a frozen prototype address-alignment test,
- and a frozen-backbone trainable residual LUT adapter study as a candidate
  route rather than the current proof.

It should not be framed as:

- a full LUT implementation of QKFormer,
- a hardware deployment paper,
- a temporal coding paper,
- or a proof that sparse LUTs reduce real energy.

## Literature Positioning

### Spiking Transformers

Use this line to justify why QKFormer is the right substrate.

- QKFormer, NeurIPS 2024, is the primary base model. Its abstract states that
  spike-form Q-K attention models token/channel attention through binary
  vectors and supports linear complexity and larger models.
  Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/179f5dcdeedc149443ebd3ba70811dbd-Abstract-Conference.html

- Spikformer introduced spike-form Q/K/V self-attention without softmax and
  showed that directly trained spiking Transformers can be competitive.
  Source: https://arxiv.org/abs/2209.15425

- Spike-driven Transformer emphasizes event-driven execution, binary spike
  communication, sparse additions, mask/add operations, and linear complexity.
  Source: https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html

Paper use:

- Safe: "Recent spiking Transformers replace dense softmax attention with
  spike-form attention modules over binary spike tensors."
- Safe: "QKFormer exposes binary Q/K activity that can be audited as an
  address space."
- Avoid: "Our method inherits the energy gains of QKFormer" unless we measure
  or use only QKFormer-reported claims as background.

### Parameter-Efficient Adapters

Use this line to justify frozen-backbone E3.

- Adapter modules freeze the original network and add a small number of
  trainable task parameters. Houlsby et al. report near full fine-tuning
  performance on GLUE while adding only 3.6% parameters per task.
  Source: https://proceedings.mlr.press/v97/houlsby19a.html

- LoRA freezes pretrained weights and injects trainable low-rank matrices.
  Source: https://arxiv.org/abs/2106.09685

Paper use:

- Safe: "E3 follows the frozen-backbone adapter paradigm, but the adapter is
  indexed by binary spiking Q/K addresses rather than by dense low-rank weight
  updates."
- Stronger novelty angle: "Unlike generic adapters, the trainable parameters
  are conditionally selected by internal spike addresses."
- Avoid: comparing directly against LoRA unless we implement a LoRA/adapter
  baseline inside QKFormer.

### Temporal Coding

Use this only as related work/future work for now.

- TTFS/rank-order/N-of-M coding are established temporal-code families in SNNs.
  Source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.971937/full

- Temporal Spike Sequence Learning via Backpropagation trains deep SNNs with
  temporal spike-sequence dependencies.
  Source: https://proceedings.neurips.cc/paper/2020/hash/8bdb5058376143fa358981954e7626b8-Abstract.html

Paper use:

- Safe: "Temporal coding suggests future extensions where lookup addresses
  include first-spike time, rank order, or spike-count features."
- Possible E4: add time-step occupancy, first-active-step, and spike-count bins
  to the QK address and compare against current binary addresses.
- Avoid: "Our current method uses relative temporal coding." Current E0-E3
  addresses are binary activation/context addresses, not verified timing codes.

### LUTs, Hashes, and Factorized Tables

Use this line to motivate scalable table design and E4, not as a proven result.

- Hash embeddings use multiple hashes into a shared embedding pool to handle
  huge vocabularies with smaller memory.
  Source: https://papers.nips.cc/paper_files/paper/2017/hash/f0f6ba4b5e0000340312d33c212c3ae8-Abstract.html

- Product quantization decomposes a vector space into a Cartesian product of
  low-dimensional subspaces and quantizes each subspace separately.
  Source: https://pubmed.ncbi.nlm.nih.gov/21088323/

- Tensor Train compression represents dense neural layers in a compact
  multilinear format.
  Source: https://arxiv.org/abs/1509.06569

Paper use:

- Safe: "Large discrete address spaces naturally raise sparse-table and
  collision issues; factorized, hashed, or compositional tables are plausible
  scalability routes."
- Possible E4: compare full address table vs token/channel-only table vs
  additive factorized subtables vs hashed table under matched parameter budget.
- Avoid: "QK-LUTFormer solves table explosion" before E4.

### FPGA LUT Neural Inference

Use this for hardware motivation, not neuromorphic energy claims.

- LUTNet uses FPGA native LUTs as inference operators.
  Source: https://arxiv.org/abs/1904.00938

- LogicNets maps quantized neural networks into truth-table netlists and notes
  that truth-table hardware cost grows exponentially with fan-in.
  Source: https://arxiv.org/abs/2004.03021

Paper use:

- Safe: "Lookup-addressed neural modules are aligned with table-based hardware
  thinking, and fan-in/address width is a first-order cost driver."
- Good proxy: table entries, active entries, address bit width, sparse storage
  ratio, and lookups per sample.
- Avoid: claiming FPGA latency/area without synthesis.

### Neuromorphic Hardware and Simulation

Use this for future deployment paths and proxy protocol.

- EBRAINS offers remote access to BrainScaleS and SpiNNaker for accepted
  research projects and describes PyNN-based workflows.
  Sources:
  https://ebrains.eu/data-tools-services/computing-infrastructure/neuromorphic-computing
  https://wiki.ebrains.eu/bin/view/Collabs/neuromorphic/Getting%20access/

- Lava can execute processes on CPU/GPU or neuromorphic hardware such as Intel
  Loihi and provides utilities for power/performance profiling.
  Source: https://lava-nc.org/

- SpikingJelly and snnTorch are practical PyTorch-based SNN simulation and
  training frameworks.
  Sources:
  https://spikingjelly.readthedocs.io/
  https://snntorch.readthedocs.io/en/latest/snntorch.html

Paper use:

- Safe: "We report hardware-oriented proxies and identify a path to Lava or
  EBRAINS evaluation."
- Avoid: "runs on Loihi/SpiNNaker/BrainScaleS" unless actually ported and
  measured.

## New Paper Story

### Problem

Softmax attention is hard to LUT-ize because pairwise Q/K addresses do not
identify row-normalized responses. The old SDR-LUTAttn results support this as
a negative motivation.

QKFormer changes the question: rather than forcing a softmax Transformer into
a lookup table, start from a spiking Transformer whose attention already uses
binary Q/K spike vectors and token/channel gating. Ask whether those internal
spike patterns are useful addresses.

### Hypothesis

The learned Q/K/gate spike patterns in early QKFormer attention blocks form a
discrete address space that can condition a tiny local adapter better than
non-address smoothing controls.

Current status is more nuanced. E0-E2 establish feasibility and limits. The
T=1 five-way trainable E3 control shows that trainable residual tables can
blur address-specific claims because a shuffled address table can partially
adapt. The T=1 frozen E2 alignment test is the strongest current support:
when prototypes are frozen, correct Q/K address lookup beats shuffled address
and global mean on Acc@1 delta, loss drift, KL, logit MSE, and local MSE.

### Main Method

The manuscript method should be separated into an address proof and an adapter
route.

Address proof:

1. Calibrate frozen response prototypes from Q/K-derived addresses.
2. Evaluate `address_lut`, `shuffled_address_lut`, and `global_mean` under the
   same calibration bank and full validation split.
3. Treat local MSE, KL, and logit MSE as primary mechanistic evidence; Acc@1 is
   downstream sanity.

Trainable E3 can become the manuscript method only if stricter controls pass:

1. Freeze trained QKFormer.
2. Extract binary Q/K/gate-derived addresses from a chosen attention module.
3. Initialize a residual LUT from calibration prototypes with count-aware
   shrinkage.
4. Insert a local adapter at `proj_lif`:

   `y_adapter = y_original + alpha * (LUT[address] - y_original)`

5. Train only LUT tables and alpha with CE + KL-to-baseline + local MSE.
6. Compare against matched controls.

Current E3 warning:

- T=1 three-way E3 suggested `address_lut` was positive against global and
  token/channel controls.
- T=1 five-way E3 showed `shuffled_address_lut` mean Acc@1 delta +0.0967,
  higher than `address_lut` +0.0600.
- Interpretation: the adapter mechanism works, but trainable tables are not
  clean address-semantic evidence unless shuffled controls are constrained or
  clearly beaten.

### Contributions

Use this contribution wording now:

1. We introduce a diagnostic framework for lookup-addressability in spiking
   Q-K attention, measuring address occupancy, hit rate, singleton buckets, and
   conditional response variance.
2. We show that forward-only address prototypes are measurable but insufficient:
   they modestly improve reconstruction, but downstream replacement is nearly
   matched by global smoothing controls.
3. We show with a frozen prototype alignment test that correct Q/K address
   lookup preserves stage1 responses better than shuffled address and global
   controls.
4. We study frozen-backbone residual LUT adapters as a candidate route and
   show why matched shuffled controls are required before claiming trainable
   address-specific gains.
5. We provide hardware-oriented proxy metrics for table size and lookup
   activity without claiming measured hardware energy.

Unsafe upgrade: do not say the trainable adapter already proves address
semantics or improves accuracy generally.

## Minimum Acceptable Experiment Package

### Must Have

1. E0 real-data trained checkpoint table:
   - per-stage coverage,
   - occupancy entropy,
   - singleton fraction,
   - hit rate,
   - conditional variance.

2. E1 reconstruction:
   - address LUT vs global mean vs candidate/background or token/channel
     coarse control,
   - stage-wise MSE reduction,
   - calibration/eval split.

3. E2 controlled replacement:
   - stage1-only, stage2-only, stage1+stage2,
   - blend 0.25 vs 1.0,
   - address vs global vs shuffled,
   - logit MSE/KL and accuracy.

4. E2 frozen alignment:
   - address_lut,
   - shuffled_address_lut,
   - global_mean,
   - same frozen prototype bank,
   - local MSE, logit KL/MSE, loss, Acc@1.

5. E3 main table:
   - address_lut,
   - global_mean,
   - token_channel_lut,
   - shuffled_address_lut if implemented for E3,
   - at least 3 seeds,
   - same train/eval protocol,
   - trainable parameters reported.

6. E3 diagnostic table:
   - alpha learned value,
   - local MSE,
   - logit KL,
   - Acc@1/Acc@5,
   - per-seed mean and standard deviation.

7. Proxy cost table:
   - address bits,
   - table entries,
   - active entries,
   - sparse storage ratio,
   - lookups per image,
   - replaced activation values,
   - original module MAC/add proxy.

### Stronger Package

1. Repeat E3 on CIFAR-100 if feasible.
2. Sweep more stage1 blocks, not just `stage1.0.tssa`.
3. Add factorized LUT E4 under same parameter budget.
4. Add time-feature E4 only if we can clearly define and validate timing
   features from QKFormer time steps.

## Reviewer-Facing Acceptance Logic

Reviewer objection: "This is just smoothing."

Required answer:

- Frozen address_lut beats global_mean and shuffled_address_lut under the same
  prototype bank on local MSE and drift metrics.
- E3 address_lut must also beat global_mean, token_channel_lut, and
  shuffled_address_lut before claiming trainable adapter address semantics.
  Current E3 five-way does not pass this bar.

Reviewer objection: "The effect is too small."

Required answer:

- Report not only accuracy delta but drift trade-off: lower KL/logit MSE at
  similar or better accuracy.
- Show stability over seeds and calibration subsets.

Reviewer objection: "CIFAR-10 is too small."

Required answer:

- Add CIFAR-100 if time permits.
- Otherwise explicitly label this as a feasibility and audit paper.

Reviewer objection: "No hardware evidence."

Required answer:

- Replace all energy claims with proxy language.
- Provide reproducible operation/table metrics and a future porting path.

Reviewer objection: "Temporal/anchor coding is invented after the fact."

Required answer:

- Keep it in related work/future work.
- Only add it to method after E4 timing-address experiments.

## Recommended Tables

Table 1: E0 address audit.

Columns:

- stage/block,
- address space,
- active entries,
- coverage,
- hit rate,
- singleton fraction,
- entropy,
- conditional variance.

Table 2: E1 split-aware reconstruction.

Columns:

- mode,
- global MSE,
- address MSE,
- relative MSE reduction,
- stage-wise reduction,
- calibration batches.

Table 3: E2 replacement and controls.

Columns:

- target,
- mode,
- blend,
- calibration setting,
- Acc@1 delta,
- loss delta,
- KL,
- logit MSE,
- local MSE.

Table 4: E3 main adapter result.

Columns:

- adapter mode,
- trainable params,
- alpha,
- Acc@1,
- Acc@1 delta,
- loss,
- KL,
- local MSE,
- mean/std across seeds.

Table 5: Proxy cost.

Columns:

- target module,
- original response shape,
- lookup count,
- address bits,
- dense table size,
- active table size,
- sparse storage ratio,
- replaced operation proxy.

## Submission Strategy

### Main-Conference Attempt

This path is credible only if all of the following hold:

1. E3 address_lut wins against global_mean, token_channel_lut, and
   shuffled_address_lut with consistent mean/std over at least 3 seeds.
2. The gain is not only Acc@1; it also improves or preserves loss/KL/local MSE
   trade-offs.
3. The method is tested beyond a single 512-image or single-seed setting.
4. The paper includes a clear negative-control story showing why E2 alone was
   insufficient.
5. There is at least one broader validation axis:
   - CIFAR-100,
   - more stage1 modules,
   - or a factorized/temporal E4 ablation.

Without item 5, the work is likely seen as a careful but narrow CIFAR-10
feasibility study.

### Workshop / Specialized Venue Attempt

This path is credible if:

1. E0-E2 are complete and well controlled.
2. E3 is either positive or honestly negative.
3. The manuscript is framed as a diagnostic study of lookup-addressability in
   spiking Q-K attention.
4. The title avoids overpromising deployment, energy, or full replacement.

Good venue style:

- neuromorphic computing workshop,
- efficient ML workshop,
- SNN/spiking workshop,
- hardware-aware ML workshop,
- negative-result-friendly workshop.

### Stop / Pivot Condition

If E3 address_lut does not beat global_mean and shuffled_address_lut, do not
write the main claim as "Q/K addresses provide useful semantics." Instead:

- mark the address-specific trainable adapter route as NO-GO for the current
  address definition;
- write the paper as a rigorous audit showing where QKFormer addresses help
  reconstruction but fail controlled adaptation;
- or pivot to E4 factorized/temporal address design before drafting a positive
  paper.

## Immediate Next Experiments

Priority 1: complete conservative E3 control sweep.

- Modes: address_lut, global_mean, token_channel_lut, shuffled_address_lut.
- Seeds: 42, 43, 44 minimum.
- Target: `stage1.0.tssa` first.
- Evaluation: full CIFAR-10 validation.
- Required metrics: Acc@1/Acc@5, loss, KL to baseline, logit MSE, local MSE,
  trainable parameters, alpha.
- Current script supports address_lut/global_mean/token_channel_lut. Add
  shuffled_address_lut to E3 before using it as a main control.

Priority 2: E3 target expansion if Priority 1 is positive.

- Targets: `stage1.0.tssa`, `stage1.1.tssa`, possibly all stage1 blocks.
- Compare single-target vs multi-target adapters.
- Keep parameter counts and alpha values explicit.

Priority 3: proxy-cost report.

- Add a small result parser that extracts active table entries, dense/sparse
  table size, average lookups, hit rate, and replaced-value counts from E0-E3
  metrics.

Priority 4: one scalability ablation.

- Choose factorized LUT first if we want a table-scaling story.
- Choose timing-address E4 first if we want to connect to temporal coding.
- Do not do both before E3 proves there is a main positive signal.

## Paper Outline

1. Introduction
   - Softmax-LUT failure motivates address-aware spiking attention.
   - QKFormer provides binary Q/K/gate attention, so the question becomes
     addressability, not conversion of dense softmax.
   - State exact claim boundary.

2. Related Work
   - Spiking Transformers.
   - ANN-to-SNN conversion and why this is not that.
   - Frozen-backbone adapters.
   - Lookup/table-based neural computation.
   - Neuromorphic simulation and hardware proxy metrics.

3. Addressability Diagnostics
   - QKFormer attention hooks.
   - Address definition.
   - Response target.
   - E0/E1 metrics.

4. Forward-Only Replacement Is Not Enough
   - E2 method.
   - Stage-wise results.
   - Blend and calibration stability.
   - Global and shuffled controls.

5. Trainable Residual LUT Adapter
   - Frozen backbone.
   - Prototype/shrinkage initialization.
   - Adapter equation.
   - Loss.
   - Controls.

6. Experiments
   - CIFAR-10 main.
   - optional CIFAR-100.
   - E0-E3 tables.
   - proxy cost.

7. Limitations
   - no real hardware,
   - no full wrapper,
   - limited datasets,
   - temporal/factorized LUT future work.

## Draft Abstract Variant

Spiking Transformers replace dense attention with binary event-driven
computation, but it remains unclear whether their internal spike patterns form
useful discrete addresses for lookup-based neural modules. We study this
question in QKFormer, a spiking Transformer whose Q-K attention operates over
binary query/key spike vectors. Instead of claiming a full LUT replacement, we
first audit the lookup-addressability of trained Q/K/gate activity through
address occupancy, hit rate, singleton buckets, and conditional response
variance. Split-aware reconstruction shows that address-conditioned prototypes
carry measurable but modest response information. Controlled replacement then
reveals a limitation: forward-only address prototypes are nearly matched by
non-address global smoothing, weakening the case for naive LUT replacement. To
test whether address semantics become useful when trained, we introduce a
frozen-backbone residual LUT adapter initialized from address prototypes and
trained with label, logit, and local-response losses. Matched controls against
global, shuffled-address, and token/channel-only tables determine whether
binary Q/K addresses provide structure beyond smoothing. This staged protocol
separates supported addressability claims from future hardware deployment
claims.
