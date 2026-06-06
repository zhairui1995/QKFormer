# QK-LUTFormer Quick State

Purpose: low-token bootstrap for continuing this project. Read this first;
open the full handoff/status docs only when details, claim boundaries, or code
history are needed.

## Current Verdict

`CONDITIONAL GO`

The strongest current signal is the CIFAR-10 T=1 conservative E3 adapter sweep:
Q/K address LUT beats both non-address controls on mean Acc@1 delta.

## Key Evidence

- T=4 QKFormer best checkpoint: 96.08% Acc@1.
- T=1 QKFormer best checkpoint: 95.20% Acc@1.
- T=1 E0: coverage 0.580482, singleton 0.079362, conditional variance
  0.054878.
- T=1 E3 multi-seed conservative sweep:
  - `address_lut`: deltas +0.12 / +0.07 / -0.01, mean +0.0600.
  - `global_mean`: deltas -0.07 / -0.17 / -0.04, mean -0.0933.
  - `token_channel_lut`: deltas -0.06 / +0.06 / -0.07, mean -0.0233.
  - Mean loss deltas: address -0.002594, global +0.000182,
    token-channel -0.001102.

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

1. Repeat T=1 E3 on another checkpoint/training seed to test checkpoint
   dependence.
2. Add `shuffled_address_lut` to E3 or the key E2/T=1 control table.
3. Broaden only after the CIFAR-10 address-specific story remains stable:
   larger dataset, harder stage, or ImageNet-lite style stress.

## Claim Boundary

Allowed now: T=1 conservative residual adapter evidence suggests Q/K binary
addresses carry useful structure beyond global smoothing and token/channel-only
addressing.

Not allowed yet: energy gain, latency gain, ImageNet gain, production LUT
wrapper, or full pure-spike Transformer claims.

