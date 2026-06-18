# QKFormer CIFAR-10 T=4 Control Replication

Protocol: seed-42 checkpoint, `stage1.0.tssa`, 128 calibration batches,
`min_support=2`, full 10,000-image validation, and calibration seeds
42/142/242/342/442. All model, LUT, and calibration hyperparameters were fixed;
only the shuffled calibration subset changed.

| Calibration seed | Aligned | Global mean | Token/channel | Aligned - global | Aligned - token | Global - token |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 95.81 | 95.63 | 95.45 | +0.18 | +0.36 | +0.18 |
| 142 | 96.02 | 95.63 | 95.41 | +0.39 | +0.61 | +0.22 |
| 242 | 95.83 | 95.63 | 95.63 | +0.20 | +0.20 | +0.00 |
| 342 | 95.74 | 95.63 | 95.65 | +0.11 | +0.09 | -0.02 |
| 442 | 95.70 | 95.63 | 95.57 | +0.07 | +0.13 | +0.06 |
| Mean | 95.82 | 95.63 | 95.542 | +0.190 | +0.278 | +0.088 |

Paired calibration-subset differences, using a two-sided t interval with
`n=5` and `df=4`:

| Contrast | Mean difference | Sample SD | 95% CI |
|---|---:|---:|---:|
| Aligned - global mean | +0.190 pp | 0.123 pp | [+0.037, +0.343] pp |
| Aligned - token/channel | +0.278 pp | 0.212 pp | [+0.014, +0.542] pp |
| Global mean - token/channel | +0.088 pp | 0.107 pp | [-0.045, +0.221] pp |

Interpretation:

- The original 0.18 pp global-mean advantage over token/channel is not a
  stable ordering. One subset ties and one reverses the ordering; its interval
  crosses zero. Treat these controls as statistically indistinguishable under
  this replication.
- Aligned lookup is higher than both controls in all five calibration subsets.
  Both paired 95% confidence intervals exclude zero. This is statistically
  consistent protocol-scoped evidence, but it is still one checkpoint and one
  fixed validation set, not independent-backbone replication.
- Global mean has lower local current MSE than aligned lookup in these runs,
  while aligned has slightly higher Acc@1. Current-replacement accuracy and
  local reconstruction fidelity therefore support different, complementary
  claims; semantic-address necessity should continue to rely primarily on the
  held-out E1/E7 reconstruction controls.
