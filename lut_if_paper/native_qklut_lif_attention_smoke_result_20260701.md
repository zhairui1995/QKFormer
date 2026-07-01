# Native QK-LUT-LIF Attention Smoke Result

Date: 2026-07-01

Status: `SMOKE-PASS` for the native attention-scoped QK-LUT-LIF training route.
This is a feasibility result only, not an accuracy or near-SOTA result.

## Protocol

- Code snapshot: public fork branch `codex/qklut-lif-rl-architecture`, runner
  commit `29f46be`.
- Dataset/model: QKFormer CIFAR-100 `T=4`, seed 42.
- Rows: original QKFormer baseline and native attention-scoped QK-LUT-LIF.
- Budget: 1 epoch, 8 training batches, 4 validation batches, batch size 16.
- Cooldown epochs: 0, so scheduled epochs equals the registered smoke budget.
- Native target scope: attention LIF nodes under `.tssa.` and `.ssa.`.
- Native LUT address precision: 6-bit state, 8-bit input.
- Static ranges: current `[-8, 8]`, state `[0, 2]`.
- Local artifacts:
  `results/native_qklut_lif_attention_c100_t4_seed42_smoke_20260701_141741/`.

## Result

| Row | Best epoch | Smoke Acc@1 | Smoke Acc@5 | Smoke loss |
|---|---:|---:|---:|---:|
| QKFormer baseline | 0 | 3.1250 | 9.3750 | 4.605469 |
| Native attention QK-LUT-LIF | 0 | 3.1250 | 9.3750 | 4.605469 |

Native QK-LUT-LIF module summary from the runner:

- replaced attention LIF targets: 18;
- table entries: 294,912;
- FP32 value footprint proxy: 1,152.0 KiB;
- trainable parameter count increases from 6,740,884 to 7,035,814.

Selected targets:

```text
stage1.0.tssa.{q_lif,k_lif,attn_lif,proj_lif}
stage2.0.tssa.{q_lif,k_lif,attn_lif,proj_lif}
stage3.0.ssa.{q_lif,k_lif,v_lif,attn_lif,proj_lif}
stage3.1.ssa.{q_lif,k_lif,v_lif,attn_lif,proj_lif}
```

## Gate

- baseline row completed: `True`;
- native row completed: `True`;
- native selected at least one attention target: `True`;
- scheduled epochs matched the registered smoke budget: `True`;
- validation ran for the fixed smoke batches: `True`;
- non-finite loss observed: `False`.

Formal smoke verdict: `SMOKE-PASS`.

## Invalid Debug Run

An earlier debug run at
`results/native_qklut_lif_attention_c100_t4_seed42_smoke_20260701_141204/`
is not used for the smoke gate because the runner had not yet overridden
`cooldown_epochs`, so the training script scheduled 11 epochs from a nominal
1-epoch request. It remains useful only as a runner-debug artifact.

## Claim Impact

This result reopens the architecture route only at the feasibility level:
native attention-scoped QK-LUT-LIF can be inserted into QKFormer and trained
end-to-end from initialization without immediate runtime or non-finite-loss
failure. It does not support a positive accuracy claim. The next registered
step is the fixed 30-epoch pilot comparing the same baseline and native rows
under identical budgets.
