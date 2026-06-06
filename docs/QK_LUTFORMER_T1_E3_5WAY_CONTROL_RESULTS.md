# QK-LUTFormer T=1 E3 5-Way Control Results

## Scope

- Setting: CIFAR-10 T=1, conservative E3 residual LUT adapter, stage1-only target.
- Five-way comparison: QKFormer baseline plus `address_lut`, `global_mean`, `token_channel_lut`, and `shuffled_address_lut` controls.
- Artifact policy: this report is based on lightweight `metrics.json`, `train_log.txt`, manifest, and summary files only; checkpoint weights are not required locally.

## Backbone And E0 Context

- T=1 train result: `qkformer_cifar10_train_20260606_001151`.
- Best validation Acc@1: 95.2% at epoch 407; final logged Acc@1: 94.64%.
- T=1 E0 result: `qkformer_lut_e0_diag_20260606_124449`.
- E0 address coverage / singleton fraction / conditional variance: 0.580482 / 0.079362 / 0.054878.

## Aggregate Results

| Setting | n | Baseline Acc@1 | Adapter Acc@1 | Delta Acc@1 | Delta Loss | KL | Logit MSE | Local MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `address_lut` | 3 | 94.6667 | 94.7267 | 0.0600 | -0.002594 | 0.037674 | 0.079307 | 0.000931 |
| `global_mean` | 3 | 94.7067 | 94.6133 | -0.0933 | 0.000182 | 0.036839 | 0.079470 | 0.001050 |
| `token_channel_lut` | 3 | 94.7400 | 94.7167 | -0.0233 | -0.001102 | 0.035720 | 0.078201 | 0.000980 |
| `shuffled_address_lut` | 3 | 94.6367 | 94.7333 | 0.0967 | -0.000587 | 0.036910 | 0.079442 | 0.001014 |

## Per-Seed Rows

| Mode | Seed | Baseline Acc@1 | Adapter Acc@1 | Delta Acc@1 | Delta Loss | Result Dir |
|---|---:|---:|---:|---:|---:|---|
| `address_lut` | 42 | 94.8800 | 95.0000 | 0.1200 | -0.005570 | `qkformer_lut_e3_trainable_lut_20260606_162845` |
| `address_lut` | 43 | 94.5800 | 94.6500 | 0.0700 | -0.001274 | `qkformer_lut_e3_trainable_lut_20260606_163043` |
| `address_lut` | 44 | 94.5400 | 94.5300 | -0.0100 | -0.000939 | `qkformer_lut_e3_trainable_lut_20260606_163244` |
| `global_mean` | 42 | 94.7900 | 94.7200 | -0.0700 | -0.000014 | `qkformer_lut_e3_trainable_lut_20260606_163446` |
| `global_mean` | 43 | 94.8100 | 94.6400 | -0.1700 | 0.003325 | `qkformer_lut_e3_trainable_lut_20260606_163740` |
| `global_mean` | 44 | 94.5200 | 94.4800 | -0.0400 | -0.002765 | `qkformer_lut_e3_trainable_lut_20260606_164038` |
| `token_channel_lut` | 42 | 94.9100 | 94.8500 | -0.0600 | -0.000549 | `qkformer_lut_e3_trainable_lut_20260606_164332` |
| `token_channel_lut` | 43 | 94.6000 | 94.6600 | 0.0600 | -0.000775 | `qkformer_lut_e3_trainable_lut_20260606_164537` |
| `token_channel_lut` | 44 | 94.7100 | 94.6400 | -0.0700 | -0.001982 | `qkformer_lut_e3_trainable_lut_20260606_164745` |
| `shuffled_address_lut` | 42 | 94.6400 | 95.0100 | 0.3700 | -0.002194 | `qkformer_lut_e3_trainable_lut_20260606_164944` |
| `shuffled_address_lut` | 43 | 94.9300 | 94.5400 | -0.3900 | 0.005606 | `qkformer_lut_e3_trainable_lut_20260606_165142` |
| `shuffled_address_lut` | 44 | 94.3400 | 94.6500 | 0.3100 | -0.005174 | `qkformer_lut_e3_trainable_lut_20260606_165343` |

## Interpretation

- Verdict: `NO-GO for address-specific accuracy claim at this setting`.
- Address-alignment check: `address_lut` mean Delta Acc@1 is 0.0600, while `shuffled_address_lut` is 0.0967.
- Claim boundary: this supports or weakens an address-specific adapter ablation only; it does not support energy, latency, ImageNet, or full-wrapper claims.

## Local Evidence Check

- Report root: `/Users/cvue/Documents/github_zr/sdr-lutattn-qkformer-lut`.
- Local artifact set should contain no checkpoint weights; verify with `find results -name '*.pth*' -o -name '*.pt' -o -name '*.ckpt'` before committing artifacts.
