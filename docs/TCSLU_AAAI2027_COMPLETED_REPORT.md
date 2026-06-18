# TCSLU AAAI2027 Completed Experiment Report

Updated: 2026-06-18

## Server Execution

Server:

- Host: `lbz@192.168.70.60`
- Repo: `/home/lbz/mac_agent/sdr-lutattn-qkformer-lut`
- Python: `/home/lbz/miniconda/envs/sdr/bin/python`
- Data root: `/home/datasets`

Completed runs:

| Run | GPU | Result directory | Rows |
|---|---:|---|---:|
| QKFormer CIFAR-10 T=1 current replacement | 0 | `results/tcslu_qkf_c10_t1_full_20260618_161423` | 7 |
| QKFormer CIFAR-10 T=4 current replacement | 1 | `results/tcslu_qkf_c10_t4_full_20260618_161423_t4rerun` | 7 |
| QKFormer CIFAR-100 T=1 current replacement | 1 | `results/tcslu_qkf_c100_t1_full_20260618_104451` | 7 |
| QKFormer CIFAR-100 T=4 current replacement | 3 | `results/tcslu_qkf_c100_t4_full_20260618_104512` | 7 |
| Spikformer-4-384w CIFAR-10 T=4 current replacement | 1 | `results/tcslu_spik_c10_t4_boundary_rerun_20260618_105951` | 6 |

Each result directory contains `summary.csv` and `metrics.json`. The result
directories were downloaded to the local `results/` tree.

## Main Results

| Architecture | Dataset | T | Static hard | Best calibrated / TCSLU row | Main finding |
|---|---|---:|---:|---:|---|
| QKFormer | CIFAR-10 | 1 | 94.64 -> 93.64 | 94.64 -> 94.57 | full-validation near-lossless current replacement |
| QKFormer | CIFAR-10 | 4 | 95.70 -> 95.08 | 95.70 -> 95.81 | calibrated current replacement is preserved; positive delta is noise-level |
| QKFormer | CIFAR-100 | 1 | 77.78 -> 75.55 | 77.78 -> 77.26 | calibration limits the drop to 0.52 pp |
| QKFormer | CIFAR-100 | 4 | 81.09 -> 68.02 | 81.09 -> 80.65 | static LUT collapses at T=4; calibration restores to 0.44 pp drop |
| Spikformer-4-384w | CIFAR-10 | 4 | 88.16 -> 66.47 | 88.16 -> 76.35 | temporal+channel calibration halves the static-LUT gap but is not lossless |

The QKFormer CIFAR-10 and CIFAR-100 rows support the TCSLU paper line:
Q/K spike-address lookup can query local current responses, and
moment/channel-style calibration restores the current contract enough to
preserve end-to-end accuracy in the completed single-layer diagnostics. The
CIFAR-10 T=4 positive delta should be described as preservation/no-loss, not as
an accuracy improvement.

The Spikformer row remains a boundary result. It supports the claim that
temporal/channel calibration is necessary and helpful, but it does not support
broad cross-architecture lossless replacement.

## CIFAR-10 T=4 Control Replication

The aligned, global-mean, and token/channel controls were repeated with five
fixed 128-batch calibration subsets while holding the checkpoint, target,
validation set, and all hyperparameters constant. Mean Acc@1 is 95.820%,
95.630%, and 95.542%, respectively.

- aligned minus global: +0.190 pp, 95% CI [+0.037, +0.343];
- aligned minus token/channel: +0.278 pp, 95% CI [+0.014, +0.542];
- global minus token/channel: +0.088 pp, 95% CI [-0.045, +0.221].

The original single-subset global-over-token ordering is therefore not
distinguishable from calibration-subset variation. Aligned lookup is higher
than both controls in all five subsets, but this remains one-checkpoint
calibration replication rather than independent-backbone evidence.

## Traceable Tables

- Machine-readable status:
  `paper/tables/data/tcslu_current_replacement_status.csv`
- Generated Markdown table:
  `paper/tables/tcslu_aaai2027_tables.md`
- JSON snapshot:
  `paper/tables/data/tcslu_aaai2027_table_snapshot.json`

Note: the newly completed QKFormer CIFAR-10 full-validation rows are recorded in
`results/EXPERIMENT_RESULTS_SUMMARY.md` and
`results/tcslu_qkf_c10_full_validation_20260618.md`. They have not yet been
promoted into `paper/tables/data/tcslu_current_replacement_status.csv` or the
generated paper table.

## Verification

Verified locally:

- QKFormer CIFAR-10 T=1 `summary.csv`: 7 rows
- QKFormer CIFAR-10 T=4 `summary.csv`: 7 rows
- QKFormer CIFAR-100 T=1 `summary.csv`: 7 rows
- QKFormer CIFAR-100 T=4 `summary.csv`: 7 rows
- Spikformer T=4 `summary.csv`: 6 rows
- All five `metrics.json` files parse as JSON
- `paper/tables/data/tcslu_aaai2027_table_snapshot.json` parses as JSON
- `tools/qkformer_lut_current_unit_probe.py` compiles
- `scripts/server/run_tcslu_aaai_matrix.sh` passes shell syntax check
- `git diff --check` passes for the touched TCSLU scripts/tables/docs

Verified on server after completion:

- no real TCSLU/QKFormer/Spikformer experiment processes remain active
- all five remote result directories contain non-empty `summary.csv` and
  `metrics.json`
- GPUs are released except unrelated existing low memory occupancy

## Claim Boundary

Supported by this experiment:

- current-level static LUT replacement is insufficient for T=4 dynamics;
- QKFormer CIFAR-10 T=1/T=4 full-validation current replacement is near-lossless
  after moment/TCSLU calibration in the single-layer diagnostic;
- QKFormer CIFAR-100 T=4 can be recovered from a 13.07 pp static-LUT drop to a
  0.44 pp calibrated/TCSLU drop;
- Spikformer T=4 benefits from learned temporal+channel calibration, improving
  from a 21.69 pp static-LUT drop to an 11.81 pp drop.

Not supported:

- measured hardware acceleration, latency, or energy savings;
- ImageNet generalization;
- broad lossless replacement across Spiking Transformer architectures;
- a completed low-rank channel-adapter or trained full TCSLU implementation
  beyond the current calibrated/gated diagnostic rows.
