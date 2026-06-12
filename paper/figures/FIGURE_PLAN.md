# QK-LUTFormer Figure Plan

Status date: 2026-06-12

Paper identity: `GO-AUDIT`, a structured Q/K lookup-addressability audit.
Figures must support address structure and claim boundaries, not a residual
adapter method claim.

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

## Figure 5: Address Component Ablation

Output:

- `paper/figures/storyboard_fig5_component_ablation.md`

Status:

- Storyboard only. No PDF is generated because component-ablation results are
  not present.

Claim supported after future result:

- Which Q/K or population address components add reconstruction signal beyond
  head/token/channel coarsening.

Claims not supported:

- Factorized LUT method superiority before E4 or component gates pass.
- Stable classification utility or hardware gains.

Reserved LaTeX snippet after data exists:

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig5_component_ablation.pdf}
  \caption{Component ablation for Q/K lookup-address construction. Each row
  uses the same E1 calibration/evaluation protocol and reports held-out
  response reconstruction relative to the global mean. The plot tests whether
  Q/K-specific address fields add reconstruction signal beyond
  head/token/channel coarsening. The shuffled full-address row preserves table
  capacity while breaking address--prototype alignment.}
  \label{fig:component-ablation}
\end{figure}
```
