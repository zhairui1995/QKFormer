# QKFormer Complete LUT Replacement Gap Plan

Updated: 2026-06-19

## Current Coverage

The evaluated QKFormer CIFAR-100 `T=1`, seed-42 checkpoint has 32 learned
convolutional/linear modules:

- 10 Q/K/V projections;
- 8 MLP projections;
- 9 patch-embedding convolutions;
- 4 attention output projections;
- 1 classifier.

All 32 outputs have been substituted simultaneously by scalar-level LUT
decompositions. The cumulative full-validation row is 77.58 -> 77.98 Acc@1
with aggregate output NRMSE 0.00013521. Under the final project criterion of
at most 0.50 percentage points paired loss, this is `PASS`.

This proves functional output-substitution fidelity for every Conv1d, Conv2d,
and Linear module in this checkpoint. It does not prove that the original
operators have been removed from execution: the diagnostic hooks still run
the original modules before replacing their outputs.

## Missing Work For A Complete QKFormer LUT Replacement Claim

### 1. Real LUT Execution Path

Replace every Conv1d, Conv2d, and Linear module with a LUT-native forward
implementation rather than a post-forward hook.

Required evidence:

- the original modules are not called, verified by execution counters or a
  profiler;
- logits and Acc@1 reproduce the admitted cumulative row;
- no hidden fallback computes the original operator.

Gate:

- paired Acc@1 loss <= 0.50 pp;
- all original Conv/Linear execution counts equal zero.

### 2. BatchNorm Folding

At evaluation time, fold every BatchNorm scale and bias into the preceding LUT
values, or replace BatchNorm with an explicit table/affine lookup.

Required evidence:

- all BN modules after covered affine operators are bypassed or removed;
- per-layer output NRMSE and end-to-end Acc@1 remain within gate;
- folded-table footprint is reported.

### 3. LIF State-Transition Replacement

Implement the membrane update, reset, and spike generation as a state-transition
lookup:

`(quantized input current, quantized membrane state) -> (next state, spike)`.

This must be tested at `T=4`; `T=1` alone does not exercise recurrent membrane
state.

Required evidence:

- state and spike agreement per time step;
- membrane-state clipping/miss rates;
- complete CIFAR-100 validation Acc@1 at `T=1` and `T=4`.

### 4. Attention Arithmetic Replacement

The following non-affine attention operations remain:

- Token-QK channel reduction `sum(q)`;
- binary gate generation and `attn * k`;
- SSA `K^T V`;
- SSA `Q (K^T V)`;
- fixed scaling by 0.125.

The two SSA matrix products are the largest remaining compute gap. They require
an explicit LUT, grouped binary accumulation, or another table-addressed
decomposition with matched controls.

Required evidence:

- local current/output error for each attention arithmetic unit;
- full-validation single-block, all-attention, and cumulative replacement;
- no original matrix multiplication executed.

### 5. Pooling, Residuals, And Reductions

Define the allowed execution model precisely.

If “complete LUT replacement” means every numerical operator is table-based,
also replace or table-encode:

- patch-stage max pooling;
- residual additions;
- spatial global average;
- temporal average;
- any nontrivial reductions used for Q/K gating.

Shape transforms such as reshape, transpose, and routing may remain as data
movement, but this exception must be explicit.

### 6. Cumulative End-To-End Wrapper

Combine the LUT-native affine path, folded BN, LIF transition tables,
attention arithmetic, pooling/reductions, and residual handling in one wrapper.

Required rows:

- CIFAR-100 `T=1`, seed 42;
- CIFAR-100 `T=4`, seed 42;
- strongest original QKFormer paired baselines;
- module-family ablations and cumulative replacement.

Primary gate:

- Acc@1 loss <= 0.50 pp for both `T=1` and `T=4`;
- zero execution of every operator claimed as replaced.

### 7. Storage Feasibility

The current direct scalar tables use 248,208 KiB of FP32 values, 9.47x the
original affine weight-value count. A credible complete method needs a compact
representation:

- FP16/INT8 values;
- factorized or grouped tables;
- shared tables across channels/layers;
- metadata-inclusive keys, counters, indices, and valid bits.

Recommended gate:

- total metadata-inclusive storage <= original model affine storage, or a
  clearly justified hardware budget;
- no material loss of the cumulative fidelity result.

### 8. Runtime Or Hardware Evidence

This is mandatory only for latency, energy, SRAM, or acceleration claims, but
not for a narrowly worded functional replacement claim.

Required evidence for an acceleration claim:

- measured or cycle-accurate lookup, accumulation, and memory traffic;
- comparison against the original QKFormer on the same device/simulator;
- complete accounting of address generation and state updates.

## Safe Claim Ladder

Current supported claim:

> All Conv1d, Conv2d, and Linear outputs in one QKFormer CIFAR-100 `T=1`
> checkpoint admit near-lossless LUT-equivalent substitution.

Claim unlocked after Steps 1--6:

> Complete functional LUT execution of the evaluated QKFormer configuration,
> with explicitly allowed routing-only operations.

Claim unlocked only after Steps 7--8:

> Compact or accelerated complete QKFormer LUT implementation.

Until the corresponding gates pass, do not use “complete QKFormer LUT
replacement” without a qualifier.
