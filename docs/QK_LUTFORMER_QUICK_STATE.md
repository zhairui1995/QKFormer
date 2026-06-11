# QK-LUTFormer Quick State

Purpose: low-token bootstrap for continuing this project. Read this first;
open the full handoff/status docs only when details, claim boundaries, or code
history are needed.

## Current Verdict

`CONDITIONAL GO` for structured addresses and constrained adaptation; `NO-GO`
for a stable address-specific accuracy-gain claim with the current adapter.

The CIFAR-10 T=1 conservative E3 adapter sweep on the first checkpoint was
positive for Q/K address LUT. An independent T=1 checkpoint repeat showed that
alpha 0.1 does not replicate as a stable positive setting. A follow-up alpha
sweep found a positive low-disturbance window at alpha 0.025 on seed 43, but
the same setting did not remain positive on the independent seed-44
checkpoint.

CIFAR-100 broadens the evidence: T=1 QKFormer reaches 77.76% Acc@1 and its
stage1 address LUT beats shuffled and token/channel controls, but global-mean
smoothing remains stronger. Split-aware reconstruction nevertheless shows a
clean cross-dataset signal, and a centered residual separates aligned from
shuffled addresses at unit scale. A residual-scale sweep does not preserve
that separation and never exceeds global smoothing. Accuracy-oriented adapter
tuning is therefore stopped; the next gate tests E1 reconstruction stability
across calibration subsets and full validation.

## Key Evidence

- T=4 QKFormer best checkpoint: 96.08% Acc@1.
- T=1 QKFormer first checkpoint: 95.20% best Acc@1.
- T=1 independent checkpoint, training seed 43: 95.04% best Acc@1.
- T=1 independent checkpoint, training seed 44: 95.08% best Acc@1.
- T=1 E0: coverage 0.580482, singleton 0.079362, conditional variance
  0.054878.
- T=1 E0 on the independent seed-43 checkpoint: coverage 0.578857,
  singleton 0.082159, conditional variance 0.049788.
- T=1 E0 on the independent seed-44 checkpoint: coverage 0.583961,
  singleton 0.080493, conditional variance 0.052902.
- T=1 E3 multi-seed conservative sweep on the first checkpoint:
  - `address_lut`: deltas +0.12 / +0.07 / -0.01, mean +0.0600.
  - `global_mean`: deltas -0.07 / -0.17 / -0.04, mean -0.0933.
  - `token_channel_lut`: deltas -0.06 / +0.06 / -0.07, mean -0.0233.
  - Mean loss deltas: address -0.002594, global +0.000182,
    token-channel -0.001102.
- T=1 E3 repeat on the independent seed-43 checkpoint:
  - `address_lut`: deltas +0.15 / -0.43 / -0.04, mean -0.1067.
  - `global_mean`: deltas -0.05 / -0.17 / +0.01, mean -0.0700.
  - `token_channel_lut`: deltas -0.17 / -0.17 / +0.08, mean -0.0867.
  - This is a NO-GO for a stable address-specific accuracy claim at alpha 0.1.
- T=1 E3 alpha sweep on the independent seed-43 checkpoint:
  - alpha 0.025:
    `address_lut` +0.14/+0.16/+0.12, mean +0.1400;
    `global_mean` mean +0.0400;
    `token_channel_lut` mean +0.0000.
    `shuffled_address_lut` -0.09/-0.18/-0.04, mean -0.1033.
  - alpha 0.05:
    address/global/token-channel means -0.0833 / -0.0633 / -0.0400.
  - alpha 0.1:
    address/global/token-channel means -0.1067 / -0.0700 / -0.0867.
- T=1 E3 alpha 0.025 on the independent seed-44 checkpoint:
  - `address_lut`: deltas -0.12 / +0.21 / -0.14, mean -0.0167.
  - `global_mean`: mean +0.0000.
  - `token_channel_lut`: mean -0.0800.
  - `shuffled_address_lut`: mean -0.0833.
  - This is a NO-GO for claiming per-checkpoint stable Acc@1 improvement.
- T=1 stage2 alpha 0.025 on the seed-43 checkpoint:
  - address/global/token-channel/shuffled mean Acc@1 deltas are
    +0.0567 / +0.1067 / -0.1733 / +0.0200.
  - Address LUT has the best mean loss delta at -0.001450, but global mean has
    the best mean Acc@1 delta. This supports stage-dependent fidelity, not an
    address-specific accuracy advantage.
- T=1 stage2 alpha 0.025 on the seed-44 checkpoint:
  - address/global/token-channel/shuffled mean Acc@1 deltas are
    -0.1033 / +0.1133 / +0.0367 / -0.0233.
  - Address LUT also has a positive mean loss delta (+0.000882), while global
    mean improves loss (-0.001570). The seed-43 stage2 loss advantage does not
    replicate.
- CIFAR-100 T=1 baseline, training seed 42:
  - Best Acc@1/Acc@5: 77.76% / 93.93% at epoch 390.
  - Final Acc@1/Acc@5: 77.40% / 94.24% at epoch 409.
- CIFAR-100 T=1 E0:
  - Overall coverage 0.590042, singleton fraction 0.076803, conditional
    variance 0.074674.
  - Stage1/stage2 coverage is approximately 1.0; stage3 coverage is 0.180817.
- CIFAR-100 T=1 E1 split-aware reconstruction:
  - Three randomized 128-batch calibration subsets, each evaluated on all 313
    validation batches, give address reductions 4.4243% / 4.4207% / 4.4292%.
  - Mean reduction is 4.4247% with standard deviation 0.0042 percentage
    points; global / candidate-background / address MSE means are 0.078528 /
    0.078496 / 0.075465.
  - Stage1/stage2/stage3 mean address reductions are 10.0388% / 3.9649% /
    1.8476%; mean evaluation address hit rate is 99.9907%.
  - Reconstruction controls over the same three subsets: token/channel LUT
    3.4246%, candidate/background 0.0356%, shuffled address -15.8216%.
  - Address exceeds token/channel by 1.0001 percentage points and beats all
    controls for every calibration seed and in every stage. This is the
    strongest current address-specific positive result.
- CIFAR-10 T=4 E1 randomized-subset controls:
  - Three randomized 128-batch calibration subsets with full validation give
    address reduction 3.7080% with standard deviation 0.0038 points.
  - Token/channel / candidate-background / shuffled reductions are 2.7941% /
    0.0329% / -17.8260%; address exceeds token/channel by 0.9139 points.
  - The same ordering holds for every seed and stage, replicating the
    address-specific reconstruction result across datasets and time steps.
- CIFAR-10 T=1 E1 checkpoint robustness:
  - Training seeds 42/43/44 address reductions are 3.7597% / 5.9361% /
    4.6503%; token/channel reductions are 2.8026% / 4.4966% / 3.6514%.
  - Address beats token/channel on every checkpoint, with mean additional
    reduction 1.1318 percentage points; shuffled controls remain strongly
    negative at -18.8859% / -17.5642% / -17.0855%.
  - This establishes the positive E1 claim across datasets, time steps,
    calibration subsets, stages, and independent training checkpoints.
- CIFAR-100 E1 calibration-size sweep:
  - Address reductions at 8/32/128/512 batches are 3.8001% / 4.2756% /
    4.4243% / 4.4674%; token/channel is 3.3315% / 3.4084% / 3.4253% /
    3.4274%.
  - Address leads at every size. Eight batches recover about 85% of the
    512-batch address gain, with 99.7952% validation address hit rate.
  - This supports calibration data efficiency, not hardware latency or energy.
- CIFAR-100 T=1 stage1 alpha 0.025:
  - address/global/token-channel/shuffled mean Acc@1 deltas are
    +0.1633 / +0.3233 / -0.1333 / -0.0933.
  - Address alignment beats same-capacity shuffled and token/channel controls,
    but does not beat generic global smoothing.
- CIFAR-100 global-plus-centered-address decomposition:
  - Centered address deltas +0.32 / +0.34 / -0.07, mean +0.1967.
  - Same-capacity centered shuffled deltas -0.15 / +0.04 / -0.40, mean
    -0.1700; the alignment gap is +0.3667 points.
  - Mean loss deltas are +0.002959 for centered address and +0.000421 for
    centered shuffled, while global mean improves loss by -0.006822.
  - This strengthens the address-alignment claim but still does not beat the
    strongest global-smoothing classification or loss baseline.
- CIFAR-100 centered address-residual scale sweep:
  - Scales 0.1 / 0.25 / 0.5 aligned mean Acc@1 deltas are +0.0333 / +0.1233 /
    +0.0767; none exceeds global mean at +0.3233.
  - Matched shuffled deltas are -0.0767 / +0.2600 / +0.0033, giving alignment
    gaps +0.1100 / -0.1367 / +0.0733.
  - The aligned-versus-shuffled ordering is not scale-stable. This is a NO-GO
    for further accuracy-oriented tuning of the current adapter.

## Baseline Meaning

- QKFormer baseline is required in every E3 row: it answers whether the adapter
  improves or preserves the original model.
- `global_mean` control answers whether generic smoothing is enough.
- `token_channel_lut` control answers whether token/channel gates are enough
  without Q/K binary address detail.
- `shuffled_address_lut` is a strong reviewer-facing control for address/table
  alignment; add it to a key table or appendix when possible.
- `global_plus_address_lut` and its same-capacity shuffled control explicitly
  test whether a zero-mean address residual adds value beyond global smoothing.
- T=1 vs T=4 is a stress setting comparison, not an adapter baseline.

## Default Commands

Run current remote loop:

```bash
bash scripts/local/run_remote_qk_lut_loop.sh
```

Summarize local results with low token output:

```bash
python3 scripts/local/analyze_qk_lut_results.py --brief
```

Package/download/analyze existing server results only:

```bash
bash scripts/local/run_remote_qk_lut_loop.sh --skip-run
```

Large training jobs should use `--kind train` so the server checks GPU idleness
before falling back to GPU 2. Small diagnostics default to GPU 2.

## Next Useful Experiments

1. Convert the E0/E1 evidence into publication figures: coverage/conditional
   variance, control MSE reduction, and calibration-size curves.
2. Reframe the paper around address structure, occupancy, and response
   reconstruction; treat small Acc@1 changes as secondary negative evidence.

## Claim Boundary

Allowed now: E0/E1 diagnostics support Q/K binary addresses as structured,
well-occupied LUT indices on CIFAR-10 and CIFAR-100; T=1 E3 supports a
low-disturbance residual-adapter route whose fidelity benefit is stage- and
checkpoint-dependent. Seed-43
stage1 shows address-specific Acc@1 separation, while seed-44 stage1 does not;
CIFAR-100 shows address alignment is meaningful relative to shuffled and
token/channel controls, but global smoothing is stronger in Acc@1 and loss.
The centered residual's aligned-versus-shuffled classification ordering is not
stable across scales, so the current adapter does not support an accuracy claim.

Not allowed yet: energy gain, latency gain, ImageNet gain, production LUT
wrapper, or full pure-spike Transformer claims.
