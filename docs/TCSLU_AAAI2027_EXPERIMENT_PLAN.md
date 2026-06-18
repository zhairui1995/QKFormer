# TCSLU AAAI2027 Experiment Package

Updated: 2026-06-18

## Research Question

Why does static semantic LUT current replacement remain near-lossless in
QKFormer T=1, while degrading severely in Spikformer T=4? Can a
**Temporal-Channel Calibrated Semantic Lookup Unit (TCSLU)** preserve both
semantic addressability and the temporal/channel contract required by spiking
Transformer dynamics?

This is a preregistered next-experiment package. It does not replace the
existing `GO-METHOD` evidence for E1/E5/E7. It defines the next paper line:
static LUT is not enough; a lookup unit for spiking Transformers must preserve
semantic Q/K addresses, support-aware fallback, and temporal/channel dynamics.

## Current Evidence Already Counted

| Evidence | Status | Paper role |
|---|---|---|
| CIFAR-100 E1/E5/E7 | complete | core semantic-address and compactness evidence |
| QKFormer CIFAR-100 T=4 matched controls | complete | secondary same-architecture downstream evidence |
| QKFormer CIFAR-10 T=1 current probe | preliminary | replacement-fidelity motivation |
| Spikformer CIFAR-10 T=4 boundary row | retained only | temporal/channel mismatch motivation |

Do not rerun broad CIFAR-100 E1/E5/E7 sweeps for this paper line. New
CIFAR-100 work should target current replacement and TCSLU ablations.

## Tables To Produce

### Table A: Current Replacement Main Table

Architectures and datasets:

- QKFormer T=1, CIFAR-10, seed 42, `stage1.0.tssa`
- QKFormer T=1, CIFAR-100, seed 42, `stage1.0.tssa`
- QKFormer T=4, CIFAR-100, seed 42, `stage1.0.tssa`
- Spikformer-4-384w T=4, CIFAR-10, SSA current

Methods:

- clean baseline
- static aligned LUT
- moment-matched LUT
- temporal gate LUT
- channel gate LUT
- TCSLU: semantic LUT + moment matching + temporal gate + channel adapter +
  support-aware backoff
- controls: shuffled address, token/channel mean, global mean

Primary metrics:

- Acc@1, Acc@5, Acc@1 drop vs clean
- local MSE, local cosine, orig/LUT RMS ratio, orig/LUT std ratio
- logit KL, logit MSE
- spike sparsity, hit rate, fallback rate, supported entries, idealized KiB

Success gate:

- QKFormer T=1: moment/TCSLU drop should remain near-lossless.
- QKFormer T=4: TCSLU should improve over static and moment LUT.
- Spikformer T=4: TCSLU should materially reduce the retained 11.88 point gap
  from 88.16% to 76.28%; full recovery is not required.

### Table B: TCSLU Component Ablation

Rows:

1. static aligned LUT
2. + moment matching
3. + temporal gate
4. + channel gate
5. + low-rank channel adapter
6. + support-aware backoff
7. full TCSLU

Report the same metrics as Table A, plus trainable parameter count for learned
gates/adapters.

### Table C: Semantic Address Ablation

Rows:

- aligned Q/K address
- shuffled address
- token/channel mean
- global mean
- same-size random table

Purpose: prove TCSLU is not just generic smoothing or capacity.

### Table D: Robustness And Generalization

Rows:

- 3 fixed 128-batch calibration subsets
- 2-3 preregistered support thresholds
- QKFormer T=1 vs T=4
- mild corruption: noise and brightness, two severities
- optional CIFAR10-DVS if the data/loader path is stable

Report hit-rate drift, fallback-rate drift, spike-rate drift, and Acc@1 drop.

### Table E: Peer Spiking Transformer Context

Use reported numbers from the original papers only for positioning. This is a
context table, not a rerun comparison. It must state that TCSLU is a
replacement/calibration unit, not a new full backbone.

## Fixed Budgets And Controls

- Main seed: 42.
- Calibration: 128 batches for main CIFAR-100 current replacement unless a
  smaller pilot is explicitly marked as pilot.
- Validation: full validation for paper rows; partial validation rows remain
  preliminary.
- Support threshold: minimum support 2 by default. Additional thresholds must
  be preregistered before launch.
- No alpha, checkpoint, stage, or seed shopping.
- Spikformer remains a boundary/motivation architecture unless TCSLU passes a
  preregistered improvement gate.

## Machine-Readable Assets

- Experiment registry: `paper/tables/data/tcslu_experiment_registry.csv`
- Current replacement status: `paper/tables/data/tcslu_current_replacement_status.csv`
- Peer comparison data: `paper/tables/data/tcslu_peer_spiking_transformer_comparison.csv`
- Generated Markdown tables: `paper/tables/tcslu_aaai2027_tables.md`
- Builder: `scripts/local/build_tcslu_aaai_tables.py`
- Server entrypoint: `scripts/server/run_tcslu_aaai_matrix.sh`
