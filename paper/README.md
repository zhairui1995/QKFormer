# QK-LUTFormer Paper Draft

This directory contains a LaTeX paper draft for the QK-LUTFormer / QKFormer-LUT
hybrid branch.

Current claim boundary:

- Supported: CIFAR-10 diagnostics, T=1 address-specific residual LUT adapter
  ablation, and comparison with global/token-channel controls.
- Not yet supported: energy, latency, ImageNet, or a full production LUT
  wrapper.

Compile from this directory with:

```bash
latexmk -pdf main.tex
```

or use any equivalent LaTeX engine available locally.
