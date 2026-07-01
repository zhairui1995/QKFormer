# Native QK-LUT-LIF QK-Contract Spikformer Smoke Result

Date: 2026-07-01

Verdict: `SMOKE-PASS`.

## Scope

This run tests whether native QK-LUT-LIF can execute inside a QK-contract
Spikformer block. It is a cross-backbone smoke test, not a broad transfer or
accuracy claim.

Run spec:
`run_specs/native_qklut_lif_spikformer_qk_contract_c10_t4_seed42_20260701.yaml`.

Public code branch:
`zhairui1995/QKFormer:codex/qklut-lif-rl-architecture`, commit `ad8f4ad`.

Remote result directory:
`/home/lbz/mac_agent/sdr-lutattn-qkformer-lut/results/native_qklut_lif_spikformer_qk_contract_attention_c10_t4_seed42_smoke_20260701_143656`.

Local small-artifact mirror:
`results/native_qklut_lif_spikformer_qk_contract_attention_c10_t4_seed42_smoke_20260701_143656/`.

## Protocol

| Field | Value |
|---|---|
| Dataset | CIFAR-10 |
| Model | Spikformer-4-384w |
| Attention contract | `qk_sum` |
| Time steps | `T=4` |
| Seed | 42 |
| Mode | smoke |
| Epochs | 1 |
| Max train batches | 8 |
| Max validation batches | 4 |
| Batch size | 16 |
| Validation batch size | 16 |
| Target scope | `attention` |

## Target Selection

The native row selected 20 attention LIF modules:

- `block.{0,1,2,3}.attn.q_lif`
- `block.{0,1,2,3}.attn.k_lif`
- `block.{0,1,2,3}.attn.v_lif`
- `block.{0,1,2,3}.attn.attn_lif`
- `block.{0,1,2,3}.attn.proj_lif`

Table footprint proxy:

| Metric | Value |
|---|---:|
| Replaced modules | 20 |
| Table entries | 327,680 |
| FP32 value proxy | 1,280.0 KiB |

## Smoke Metrics

| Row | Best epoch | Best Acc@1 |
|---|---:|---:|
| QK-contract Spikformer baseline | 0 | 10.9375 |
| Native QK-LUT-LIF QK-contract Spikformer | 0 | 10.9375 |

The smoke numbers are not accuracy evidence. They only show that the fixed
cross-backbone smoke path runs, selects attention targets, keeps finite losses,
and writes matched manifest artifacts.

## Gate Decision

Smoke gate:

- both rows completed: pass;
- native variant selected attention LIF targets: pass;
- matched `comparison.json` was produced: pass;
- checkpoint manifests and summaries were produced for both rows: pass.

Overall: `SMOKE-PASS`.

## Next Action

Do not launch a Spikformer pilot before interpreting the two active QKFormer
native pilots. If QKFormer native training gives a positive or borderline
architecture signal, this Spikformer QK-contract route is the first bounded
cross-backbone transfer substrate.
