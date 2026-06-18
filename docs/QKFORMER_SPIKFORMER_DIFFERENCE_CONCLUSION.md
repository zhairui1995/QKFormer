# Why QKFormer And Spikformer Differ So Much

Updated: 2026-06-17

## Evidence Snapshot

| Architecture | Dataset | T | Injection target | Best diagnostic row | Clean Acc@1 | Injected Acc@1 | Drop |
|---|---|---:|---|---|---:|---:|---:|
| QKFormer | CIFAR-10 | 1 | `stage1.0.tssa` current before original BN/LIF | moment-matched hard current LUT | 94.63% | 94.58% | 0.05 pp |
| Spikformer-4-384w | CIFAR-10 | 4 | SSA current before original BN/LIF | learned temporal+channel gate | 88.16% | 76.28% | 11.88 pp |

QKFormer result path:
`results/qkformer_current_unit_probe_20260617_125153_valid/`.

Spikformer active boundary row:
`results/spikformer_boundary_result_20260617/summary.csv`.
Detailed Spikformer diagnostics are archived at
`results/spikformer_experiment_archive_20260617.zip`.

## Audited Conclusion

The gap is real, but it should not be framed as "QKFormer is solved and
Spikformer is invalid." The supported conclusion is narrower: the current
lookup-current unit is well matched to QKFormer T=1 single-step Q/K current
replacement, while Spikformer T=4 exposes a temporal/channel dynamics mismatch
that this static LUT design has not solved.

The most likely causes are:

1. **Time-step contract:** the QKFormer probe is T=1, so lookup-current error is
   injected once and does not accumulate through multi-step membrane dynamics.
   Spikformer is T=4; scale, sign, and channel errors pass through BN/LIF state
   across multiple steps, making small current errors much more destructive.
2. **Representation fit:** QKFormer TSSA exposes a more discrete Q/K-driven
   address space that matches subspace-decoupled lookup. Spikformer SSA appears
   to retain more continuous, cross-channel, and time-coupled current geometry.
3. **Channel geometry:** QKFormer moment matching nearly closes the hard
   replacement gap, which suggests the LUT keeps much of the current direction
   and mainly needs mean/variance correction. Spikformer still drops from
   88.16% to 76.28% even with learned temporal+channel gating, so mean/variance
   and simple channel gates are insufficient there.
4. **Evidence scale:** the QKFormer 94.63% to 94.58% result is a strong
   preliminary fidelity signal, but it is still single seed, single layer,
   CIFAR-10, T=1, and about 2K validation images. The alpha-0.25 improvement
   should be treated as within no-loss fluctuation, not as an accuracy-gain
   claim.

## Claim Boundary

Use Spikformer as a boundary/discussion result, not as a parallel positive
mainline. It shows that T>1 Spiking Transformer replacement needs a richer
temporal/channel calibration contract before hard lookup-current substitution
can be claimed as general.

Do not claim measured O(1) acceleration, energy saving, SRAM viability, or
production attention replacement from these hook diagnostics. The hooks verify
representation fidelity; they do not yet remove all original compute in a
measured deployment path.

For naming, prefer **Semantic Lookup Spiking Unit (SLSU)** or a stricter
**Semantic Lookup Current Unit**. Avoid making "Dynamics-Aware" the primary
claim until T>1 dynamics are actually recovered.

## Next Evidence Needed

Before a strong venue claim, the QKFormer line needs full-validation matched
controls, CIFAR-100 replication, multi-seed or checkpoint confirmation, more
than one target layer/stage, and either T>1 QKFormer or a clearly scoped
discussion of why the method is QKFormer-specific. A systems claim additionally
requires operation accounting that includes lookup keys, indices, fallback, and
metadata, not just prototype value footprint.
