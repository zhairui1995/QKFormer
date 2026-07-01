# LL-ViT-Inspired MLP-Scope Native QK-LUT-LIF Design

Date: 2026-07-01

## Decision

Add a bounded MLP-scope native QK-LUT-LIF experiment on QKFormer CIFAR-100
`T=4`, seed 42. This is an insertion-site test motivated by LL-ViT's channel
mixer observation, not an LL-ViT reproduction.

## Why This Route

The completed frozen-backbone LUT-LIF, CNL-LUT-LIF, RL selector, trainable CNL,
and structured-residual probes are all `NO-GO` for a positive trainable
LUT-LIF architecture claim. The current attention-scope native training route
tests whether adapting the whole network from initialization can make
QK-LUT-LIF viable.

LL-ViT suggests a different insertion-site hypothesis: transformer channel
mixers/MLPs may be a more natural location for learned LUT modules than
attention internals. In QKFormer, the matching low-risk test is to replace only
the MLP LIF nodes with trainable dense LUT-LIF neurons and keep the rest of the
training protocol matched.

## Fixed Protocol

Run spec:
`run_specs/native_qklut_lif_mlp_c100_t4_seed42_20260701.yaml`.

Runner:
`scripts/server/run_native_qklut_lif_cifar100.sh` with
`NATIVE_QKLUT_LIF_TARGET_SCOPE=mlp`.

Rows:

| Row | Scope | Purpose |
|---|---|---|
| `baseline_qkformer` | none | matched training baseline |
| `native_qklut_lif_mlp` | `mlp` | LL-ViT-inspired channel-mixer/LIF insertion |

Budgets:

| Gate | Epochs | Train batches | Eval batches | Batch | Purpose |
|---|---:|---:|---:|---:|---|
| Smoke | 1 | 8 | 4 | 16 | server feasibility and target selection |
| Pilot | 30 | full | full | 64 | insertion-site viability |
| Full | 400 | full | full | 64 | only if pilot passes |

## Success Gate

Smoke passes only if both rows finish without non-finite loss, the MLP-scope
variant selects at least one MLP LIF target, and validation executes at the
fixed smoke budget.

Pilot passes only if the MLP-scope native variant is no worse than 1.00 pp
below the matched QKFormer baseline under the same seed, data, augmentation,
epoch, batch, and validation budget.

## Claim Boundary

Allowed after smoke pass:

> MLP-scope native LUT-LIF insertion is executable in QKFormer and can be
> evaluated under a matched pilot protocol.

Allowed after pilot pass:

> LL-ViT-inspired channel-mixer placement is a credible insertion site for
> native LUT-LIF training in this QKFormer CIFAR-100 setting.

Disallowed without further evidence:

- LL-ViT reproduction;
- superiority over LL-ViT, QKFormer, Spikformer, or SpikeFormer;
- broad cross-architecture transfer;
- hardware speed, energy, area, SRAM, or latency;
- positive RL-guided policy-search claims.

## Status

Smoke completed and passed on server-lbz at commit `9a38444`. See
`lut_if_paper/native_qklut_lif_mlp_smoke_result_20260701.md`.

The attention-scope native QK-LUT-LIF pilot remains the active A-line
experiment; this MLP-scope route should not trigger a full run before its own
pilot gate passes.

P1 pilot launch update: the fixed 30-epoch MLP-scope pilot is now running on
server-lbz from public branch commit `cdef659`. Remote result directory:
`/home/lbz/mac_agent/sdr-lutattn-qkformer-lut/results/native_qklut_lif_mlp_c100_t4_seed42_pilot_20260701_143002`.
This pilot is a bounded insertion-site comparison against the attention-scope
native route; it is not a hyperparameter, seed, or bit-width sweep.
