# QK-LUTFormer Quick State

Purpose: low-token bootstrap for continuing this project. Read this first;
open the full handoff/status docs only when details, claim boundaries, or code
history are needed.

## Current Verdict

`CONDITIONAL GO`, but the accuracy-gain claim is now weaker.

The CIFAR-10 T=1 conservative E3 adapter sweep on the first checkpoint was
positive for Q/K address LUT. An independent T=1 checkpoint repeat showed that
alpha 0.1 does not replicate as a stable positive setting, but a follow-up
alpha sweep found a stable low-disturbance window at alpha 0.025.

## Key Evidence

- T=4 QKFormer best checkpoint: 96.08% Acc@1.
- T=1 QKFormer first checkpoint: 95.20% best Acc@1.
- T=1 independent checkpoint, training seed 43: 95.04% best Acc@1.
- T=1 E0: coverage 0.580482, singleton 0.079362, conditional variance
  0.054878.
- T=1 E0 on the independent seed-43 checkpoint: coverage 0.578857,
  singleton 0.082159, conditional variance 0.049788.
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

1. Update/keep the paper tables around alpha 0.025 as the main E3 result and
   alpha 0.05/0.1 as drift controls.
2. Broaden only after the CIFAR-10 low-disturbance adapter story remains stable:
   larger dataset, harder stage, or ImageNet-lite style stress.
3. Consider a third independent T=1 checkpoint or a CIFAR-100/ImageNet-lite
   smoke before making broad generalization claims.

## Claim Boundary

Allowed now: E0/E1 diagnostics support Q/K binary addresses as structured,
well-occupied LUT indices; T=1 E3 alpha sweep supports a low-disturbance
residual-adapter route at alpha 0.025 against global, token/channel, and
same-capacity shuffled-address controls.

Not allowed yet: energy gain, latency gain, ImageNet gain, production LUT
wrapper, or full pure-spike Transformer claims.
