# QK-LUTFormer CIFAR-100 TBD Execution Package

## Material Passport

- Material ID: `qk-lutformer-cifar100-tbd-20260620`
- Type: code experiment plan and executable matrix
- Status: runner ready; paper results pending full-validation execution
- Method: QK-LUTFormer / TCSLU
- Frozen backbone: QKFormer CIFAR-100, seed 42
- Dataset: CIFAR-100
- Target: `stage1.0.tssa`
- Time steps: `T=1`

## Method Boundary

QKFormer supplies the trained spiking backbone and binary Q/K states.
QK-LUTFormer is the evaluated method: it calibrates Q/K-addressed response
tables, applies matched controls, and injects LUT predictions at the selected
current boundary while preserving the downstream BN/LIF dynamics. The current
hook diagnostic does not remove the original convolutional computation and is
therefore a fidelity experiment, not a measured speedup.

## Registered Experiment Groups

| Case | Hypothesis | Strongest matched control | Fixed budget | Success/failure gate |
|---|---|---|---|---|
| `component` | temporal/channel calibration improves static Q/K LUT fidelity | static aligned LUT | 128 calibration batches; full validation | report every registered component; no post-hoc selection |
| `semantic_controls` | aligned Q/K semantics outperform capacity/smoothing explanations | shuffled, token/channel, global, same-size random | identical checkpoint, target, calibration and evaluation | aligned improves local fidelity and is not worse in Acc@1 drop |
| `calibration_subsets` | the calibrated method is not dependent on one calibration ordering | fixed seeds 42/43/44 | 128 batches per seed; full validation | report mean and worst case; no seed selection |
| `support_thresholds` | support-aware behavior remains interpretable across thresholds | thresholds 1/2/4 under the same method | fixed seed and calibration budget | report drop, hit and fallback tradeoff for all thresholds |
| `mild_corruption` | calibrated lookup degrades more gracefully than static lookup | clean validation plus fixed noise/brightness severities | severity 1/2; fixed corruption seed | report accuracy and lookup-drift metrics without severity tuning |

## Execution

The runner is:

```bash
bash scripts/server/run_qk_lutformer_cifar100_tbd_matrix.sh <case>
```

Each run writes:

- per-run `summary.csv`, `metrics.json`, and `run_metadata.json`;
- aggregate `matrix_summary.csv`;
- aggregate `matrix_manifest.json`.

The `smoke` case is an engineering check only. It may use a deliberately short
backbone-training checkpoint and must never be promoted into a paper table.
Paper rows require the fixed seed-42 trained checkpoint, 128 calibration
batches, and full CIFAR-100 validation.
