# QK-LUTFormer Figure Plan

Status date: 2026-06-14

Paper identity: scoped `GO-METHOD` for address reconstruction, explicit
backoff, and compact subspace lookup. Figures must not imply stable
classification or measured hardware gains.

## Figure 1: Audit Overview

Output:

- `paper/figures/fig1_audit_overview.pdf`

Claim supported:

- The paper evaluates lookup-addressability through a staged protocol:
  spiking Q/K states, binary address construction, local response
  reconstruction, negative controls, and explicit claim boundaries.

Claims not supported:

- Full LUT QKFormer replacement.
- Energy, latency, hardware acceleration, or ImageNet generalization.
- Stable downstream accuracy gains.

Caption draft:

`Overview of the Q/K lookup-addressability audit. Binary Q/K states and
token/channel context are packed into deterministic lookup addresses, calibrated
against local module responses, and evaluated under four controls: global mean,
coarse token/channel lookup, correct Q/K address lookup, and shuffled-address
lookup. The audit separates supported response-structure evidence from
unsupported downstream and hardware claims.`

LaTeX snippet:

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig1_audit_overview.pdf}
  \caption{Overview of the Q/K lookup-addressability audit. Binary Q/K states
  and token/channel context are packed into deterministic lookup addresses,
  calibrated against local module responses, and evaluated under four controls:
  global mean, coarse token/channel lookup, correct Q/K address lookup, and
  shuffled-address lookup. The audit separates supported response-structure
  evidence from unsupported downstream and hardware claims.}
  \label{fig:audit-overview}
\end{figure*}
```

## Figure 2: Address Controls Protocol

Output:

- `paper/figures/fig2_address_controls.pdf`

Claim supported:

- Correct-address lookup and shuffled-address lookup answer different
  questions: correct alignment tests address semantics, while shuffled
  alignment preserves table capacity but breaks address/prototype matching.
- Token/channel and global controls remove fine Q/K detail.

Claims not supported:

- That the correct-address control improves classification.
- That token/channel control is an unfair or weak baseline.

Caption draft:

`Negative-control protocol for address semantics. Correct-address lookup
retrieves the response prototype calibrated for the same Q/K-derived address,
whereas the shuffled control preserves table values and capacity but permutes
the address--prototype mapping. Coarse token/channel, candidate/background, and
global controls test whether reconstruction can be explained without fine Q/K
address detail.`

LaTeX snippet:

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig2_address_controls.pdf}
  \caption{Negative-control protocol for address semantics. Correct-address
  lookup retrieves the response prototype calibrated for the same Q/K-derived
  address, whereas the shuffled control preserves table values and capacity but
  permutes the address--prototype mapping. Coarse token/channel,
  candidate/background, and global controls test whether reconstruction can be
  explained without fine Q/K address detail.}
  \label{fig:address-controls}
\end{figure*}
```

## Figure 3: E1 Reconstruction Results

Outputs:

- `paper/figures/fig3_e1_reconstruction.pdf`
- `paper/figures/data/fig3_e1_reconstruction_data.csv`
- `paper/figures/data/fig3_e1_reconstruction_data.json`

Data sources:

- `results/qkformer_lut_e1_recon_20260611_151336_c10_t1_ckpt42_controls/metrics.json`
- `results/qkformer_lut_e1_recon_20260611_150653_c10_t1_ckpt43_controls/metrics.json`
- `results/qkformer_lut_e1_recon_20260611_150654_c10_t1_ckpt44_controls/metrics.json`
- `results/qkformer_lut_e1_recon_20260611_145102_c100_stability_seed42/metrics.json`
- `results/qkformer_lut_e1_recon_20260611_145103_c100_stability_seed43/metrics.json`
- `results/qkformer_lut_e1_recon_20260611_145104_c100_stability_seed44/metrics.json`

Claim supported:

- Correct Q/K addresses improve held-out local response reconstruction beyond
  token/channel coarsening across three CIFAR-10 checkpoints and three
  CIFAR-100 calibration seeds.
- Shuffled-address reconstruction is worse than the global mean, so table
  capacity alone does not explain the effect.

Claims not supported:

- Stable top-1 accuracy gain.
- Hardware efficiency.
- Generalization beyond the reported CIFAR settings.

Caption draft:

`E1 held-out response reconstruction under address controls. (a) Correct Q/K
addresses consistently reduce MSE more than coarse token/channel lookup across
three CIFAR-10 checkpoints and three CIFAR-100 calibration seeds. (b) Shuffling
the address--prototype alignment makes reconstruction worse than the global
mean baseline, showing that the result is not explained by table capacity
alone. Values are relative MSE reductions against the global mean and are read
directly from result JSON files.`

LaTeX snippet:

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig3_e1_reconstruction.pdf}
  \caption{E1 held-out response reconstruction under address controls. (a)
  Correct Q/K addresses consistently reduce MSE more than coarse token/channel
  lookup across three CIFAR-10 checkpoints and three CIFAR-100 calibration
  seeds. (b) Shuffling the address--prototype alignment makes reconstruction
  worse than the global mean baseline, showing that the result is not explained
  by table capacity alone. Values are relative MSE reductions against the
  global mean and are read directly from result JSON files.}
  \label{fig:e1-reconstruction}
\end{figure*}
```

## Figure 4: Calibration-Size Sweep

Outputs:

- `paper/figures/fig4_calibration_sweep.pdf`
- `paper/figures/data/fig4_calibration_sweep_data.csv`
- `paper/figures/data/fig4_calibration_sweep_data.json`

Data sources: calibration seeds 42--44 at 8, 32, 128, and 512 batches. The
exact twelve result JSON paths are recorded in
`figures/data/fig4_calibration_sweep_data.csv`.

Claim supported:

- Across CIFAR-100 calibration seeds 42--44, address reconstruction remains
  above token/channel reconstruction from 8 to 512 calibration batches.
- Validation hit rate is already high with 8 calibration batches and approaches
  saturation as calibration grows.

Claims not supported:

- Any measured memory, latency, energy, or hardware efficiency claim.

Caption draft:

`CIFAR-100 calibration-size sweep for E1 reconstruction. (a) Correct Q/K
addresses remain above token/channel lookup from 8 to 512 calibration batches.
(b) The validation address hit rate is high even at 8 batches and saturates
with more calibration data. Curves show means over calibration seeds 42--44;
error bars show population standard deviations.`

LaTeX snippet:

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig4_calibration_sweep.pdf}
  \caption{CIFAR-100 calibration-size sweep for E1 reconstruction. (a)
  Correct Q/K addresses remain above token/channel lookup from 8 to 512
  calibration batches. (b) The validation address hit rate is high even at 8
  batches and saturates with more calibration data. Curves show means over
  calibration seeds 42--44; error bars show population standard deviations.}
  \label{fig:calibration-sweep}
\end{figure}
```

## Figure 5: E7 Component And Entry-Budget Ablation

Output:

- `paper/figures/fig5_e7_component_ablation.pdf`
- `paper/figures/data/fig5_e7_component_ablation_data.json`
- `paper/figures/data/fig5_e7_component_ablation_data.csv`

Status:

- Implemented from the fixed CIFAR-100 T=1 seed-42, 128-calibration-batch,
  full-validation E7 result.

Claims supported:

- TC+Q/gate and TC+QK retain most full-address reconstruction gain.
- Both subspace variants remain below the 25% compact-address entry gate.
- Shuffled-subspace controls break the reconstruction benefit.

Claims not supported:

- Stable classification utility.
- Measured SRAM, latency, area, energy, or production hardware gains.

LaTeX snippet:

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig5_e7_component_ablation.pdf}
  \caption{Seed-42 E7 component ablation. Aligned semantic subtables retain
  most full-address gain below the 25\% entry gate, whereas shuffled-subspace
  controls are worse than the global mean.}
  \label{fig:e7-component}
\end{figure}
```
