# QK-LUTFormer Quick State

Purpose: low-token bootstrap for continuing this project. Read this first;
open the full handoff/status docs only when details, claim boundaries, or code
history are needed.

## Current Verdict

`CONDITIONAL GO`, but the accuracy-gain claim is not checkpoint-stable.

The CIFAR-10 T=1 conservative E3 adapter sweep on the first checkpoint was
positive for Q/K address LUT. An independent T=1 checkpoint repeat showed that
alpha 0.1 does not replicate as a stable positive setting. A follow-up alpha
sweep found a positive low-disturbance window at alpha 0.025 on seed 43, but
the same setting did not remain positive on the independent seed-44
checkpoint.

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

## Baseline Meaning

- QKFormer baseline is required in every E3 row: it answers whether the adapter
  improves or preserves the original model.
- `global_mean` control answers whether generic smoothing is enough.
- `token_channel_lut` control answers whether token/channel gates are enough
  without Q/K binary address detail.
- `shuffled_address_lut` is a strong reviewer-facing control for address/table
  alignment; add it to a key table or appendix when possible.
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

1. Finish the seed-44 stage2 repeat to separate layer effects from checkpoint
   variance.
2. Finish the CIFAR-100 T=1 baseline and run the prepared E0/E3 four-control
   workflow before making any cross-dataset claim.
3. Reframe the main paper claim around address structure and constrained
   fidelity; treat small Acc@1 changes as secondary and checkpoint-dependent.

## Claim Boundary

Allowed now: E0/E1 diagnostics support Q/K binary addresses as structured,
well-occupied LUT indices; T=1 E3 supports a low-disturbance residual-adapter
route whose fidelity benefit is stage- and checkpoint-dependent. Seed-43
stage1 shows address-specific Acc@1 separation, while seed-44 stage1 does not;
seed-43 stage2 favors address LUT in loss but global mean in Acc@1.

Not allowed yet: energy gain, latency gain, ImageNet gain, production LUT
wrapper, or full pure-spike Transformer claims.
