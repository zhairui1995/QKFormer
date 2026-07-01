# RL-Guided QK-LUT-LIF Policy Probe Result

Date: 2026-07-01

Status: `NO-GO` for the restricted RL/bandit insertion-policy route.

## Protocol

- Host: server-lbz, single RTX 4090 GPU.
- Code snapshot: public fork branch `codex/qklut-lif-rl-architecture`, fixed
  runner commit `b870707`.
- Dataset/model: QKFormer CIFAR-100 `T=4`, seed 42.
- Checkpoint: retained QKFormer CIFAR-100 `T=4` seed-42 checkpoint.
- Calibration: 128 train batches.
- Search: one coordinate-bandit round over 35 LIF groups.
- Actions per group: `posthoc`, `quant`, `cnl`.
- Final evaluation: full CIFAR-100 validation set.

## Result

| Method | Acc@1 | Drop | Logit MSE/sample | Reward |
|---|---:|---:|---:|---:|
| full all-posthoc | 79.55 | 1.47 | 18.355140 | -1.653551 |
| full all-quantized arithmetic | 79.61 | 1.41 | 18.291697 | -1.592917 |
| full all-CNL-LUT-LIF | **79.77** | **1.25** | 18.350349 | **-1.433503** |
| learned policy | 79.56 | 1.46 | 18.353824 | -1.643538 |

Gate:

- beats best fixed-control reward: `False`;
- paired drop <= 1.50 pp: `True`;
- all selected replacement modules executed: `True`.

Formal verdict: `NO-GO`.

## Interpretation

The restricted policy-search route does not improve the QK-LUT-LIF story. The
learned mixed policy underperforms the fixed all-CNL-LUT-LIF control on the
full-validation gate. This invalidates the simple claim that an RL/bandit
selector over existing replacement variants is enough to create a stronger new
architecture.

The result does not invalidate a trained architecture route. It says the next
credible route must change the module or training objective, not merely choose
among fixed post-hoc replacement variants.

## Invalid Prior Run

The earlier full run at
`results/rl_qklut_lif_policy_cifar100_t4_seed42_full_20260701_102538/` is
marked invalid for formal evidence because its fixed-policy controls were
evaluated on the search subset while the learned policy was evaluated on full
validation. It may be used only as a runner-debug artifact.
