# Storyboard: Figure 5 Component Ablation

Status: storyboard only. Do not generate a PDF until the component-ablation
experiment returns machine-readable results.

## Purpose

Figure 5 should answer a reviewer-facing question:

> Which address components carry the extra reconstruction signal beyond
> token/channel coarsening?

It must not imply that E4 or residual adapters are the main contribution.

## Required Data

Use E1-style held-out reconstruction results with the same calibration/eval
protocol and the following rows:

1. `head_token_channel`: head + token + channel only.
2. `plus_q_gate`: head + token + channel + Q/gate.
3. `plus_k`: head + token + channel + Q/gate + K.
4. `full_qk_population`: full Q/K/population address.
5. `shuffled_full`: same-capacity shuffled full address.

Required metrics:

- global mean MSE;
- control MSE for each component setting;
- relative MSE reduction vs global mean;
- evaluation hit rate;
- calibration batches and seed;
- dataset/checkpoint identifier;
- source result directory.

## Proposed Layout

Use a two-panel vector PDF after results exist:

- (a) Stepwise component gains: horizontal or grouped bars showing relative MSE
  reduction for each component setting.
- (b) Incremental gain over token/channel: a lollipop plot showing
  `component_reduction - token_channel_reduction`.

Color convention:

- token/channel: orange;
- Q/gate and K additions: blue gradient or two distinct blue tones;
- full address: dark blue;
- shuffled full: red.

## Caption Draft

`Component ablation for Q/K lookup-address construction. Each row uses the same
E1 calibration/evaluation protocol and reports held-out response reconstruction
relative to the global mean. The plot tests whether Q/K-specific address fields
add reconstruction signal beyond head/token/channel coarsening. The shuffled
full-address row preserves table capacity while breaking address--prototype
alignment.`

## Claim Boundary

Supported if the experiment passes:

- Q/K-specific components add measurable held-out response information beyond
  token/channel coarsening.

Not supported:

- stable classification improvement;
- factorized LUT method superiority;
- energy, latency, hardware acceleration, or full QKFormer replacement.
