# Native QK-LUT-LIF MLP-Scope Smoke Result

Date: 2026-07-01

Verdict: `SMOKE-PASS`.

## Scope

This run tests an LL-ViT-inspired insertion-site hypothesis: replace only
QKFormer MLP-scope LIF nodes with native trainable dense LUT-LIF modules. It
does not reproduce LL-ViT and does not test cross-architecture superiority.

Run spec:
`run_specs/native_qklut_lif_mlp_c100_t4_seed42_20260701.yaml`.

Public code branch:
`zhairui1995/QKFormer:codex/qklut-lif-rl-architecture`, commit `9a38444`.

Remote result directory:
`/home/lbz/mac_agent/sdr-lutattn-qkformer-lut/results/native_qklut_lif_mlp_c100_t4_seed42_smoke_20260701_142512`.

Local small-artifact mirror:
`results/native_qklut_lif_mlp_c100_t4_seed42_smoke_20260701_142512/`.

## Protocol

| Field | Value |
|---|---|
| Dataset | CIFAR-100 |
| Model | QKFormer |
| Time steps | `T=4` |
| Seed | 42 |
| Mode | smoke |
| Epochs | 1 |
| Max train batches | 8 |
| Max eval batches | 4 |
| Batch size | 16 |
| Validation batch size | 16 |
| Target scope | `mlp` |

## Target Selection

The native MLP-scope variant selected eight LIF modules:

- `stage1.0.mlp.mlp1_lif`
- `stage1.0.mlp.mlp2_lif`
- `stage2.0.mlp.mlp1_lif`
- `stage2.0.mlp.mlp2_lif`
- `stage3.0.mlp.mlp1_lif`
- `stage3.0.mlp.mlp2_lif`
- `stage3.1.mlp.mlp1_lif`
- `stage3.1.mlp.mlp2_lif`

Table footprint proxy:

| Metric | Value |
|---|---:|
| Replaced modules | 8 |
| Table entries | 131,072 |
| FP32 value proxy | 512.0 KiB |
| Baseline trainable parameters | 6,740,884 |
| Native MLP-scope trainable parameters | 6,871,964 |

## Smoke Metrics

| Row | Best epoch | Best Acc@1 | Best Acc@5 | Best loss |
|---|---:|---:|---:|---:|
| Baseline QKFormer | 0 | 3.125 | 9.375 | 4.60546875 |
| Native QK-LUT-LIF MLP | 0 | 3.125 | 9.375 | 4.60546875 |

The smoke numbers are not accuracy evidence. They only show that the fixed
smoke path runs, selects MLP targets, keeps finite losses, and writes matched
summary artifacts.

## Gate Decision

Smoke gate:

- both rows completed: pass;
- scheduled epochs fixed at 1: pass;
- fixed smoke train/eval budgets were passed to the runner: pass;
- native variant selected MLP LIF targets: pass;
- matched `comparison.json` was produced: pass.

Overall: `SMOKE-PASS`.

## Next Action

Do not launch a full run from this smoke alone. The active A-line
attention-scope native QK-LUT-LIF pilot should finish first. If that pilot is
negative or ambiguous, this MLP-scope route is the next bounded pilot because
it tests the LL-ViT channel-mixer placement hypothesis under the same
QKFormer/CIFAR-100/T=4/seed-42 protocol.
