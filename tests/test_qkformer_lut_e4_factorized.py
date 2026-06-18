from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from qkformer_lut_e3_trainable_lut import TrainableLUTModule  # noqa: E402


class FakePrototype:
    def __init__(self, kind: str = "token_qk") -> None:
        self.name = "stage1.0.tssa" if kind == "token_qk" else "stage3.0.ssa"
        self.kind = kind
        self.stage = "stage1" if kind == "token_qk" else "stage3"
        detail = 4 if kind == "token_qk" else 4 * 4 * 4
        self.address_space = 8 * 8 * 8 * detail
        self.count = torch.arange(1, self.address_space + 1, dtype=torch.float64)
        means = torch.linspace(-1.0, 1.0, self.address_space, dtype=torch.float64)
        self.sum = means * self.count
        self.global_mean = float(self.sum.sum() / self.count.sum())
        self.address_mean = means


def build_module(mode: str, kind: str = "token_qk", alpha: float = 0.025) -> TrainableLUTModule:
    return TrainableLUTModule(
        FakePrototype(kind),
        mode=mode,
        alpha_init=alpha,
        learn_alpha=False,
        shrinkage_tau=0.0,
        address_scale=1.0,
        mode_seed=42,
        token_bins=8,
        channel_bins=8,
        population_bins=4,
    )


class FactorizedLUTTests(unittest.TestCase):
    def test_address_split_round_trip_for_both_attention_kinds(self) -> None:
        for kind in ("token_qk", "spiking_self"):
            module = build_module("factorized_sum_lut", kind=kind)
            addresses = torch.arange(module.address_space, dtype=torch.long)
            components = module.split_address(addresses)
            rebuilt = torch.zeros_like(addresses)
            for name, size in zip(module.component_names, module.component_sizes):
                rebuilt = rebuilt * size + components[name]
            self.assertTrue(torch.equal(rebuilt, addresses))

    def test_matched_control_covers_factorized_gated_parameter_budget(self) -> None:
        gated = build_module("factorized_gated_lut")
        matched = build_module("matched_param_address_lut")
        gated_params = sum(parameter.numel() for parameter in gated.parameters() if parameter.requires_grad)
        matched_params = sum(parameter.numel() for parameter in matched.parameters() if parameter.requires_grad)
        self.assertEqual(gated_params, matched_params)

    def test_factorized_shuffle_preserves_capacity_and_changes_alignment(self) -> None:
        aligned = build_module("factorized_sum_lut")
        shuffled = build_module("factorized_shuffled_lut")
        self.assertEqual(aligned.summary()["table_entries"], shuffled.summary()["table_entries"])
        addresses = torch.arange(256, dtype=torch.long)
        response = torch.zeros_like(addresses, dtype=torch.float32)
        aligned_pred, _ = aligned(addresses, response)
        shuffled_pred, _ = shuffled(addresses, response)
        self.assertFalse(torch.equal(aligned_pred, shuffled_pred))

    def test_alpha_zero_is_exact_identity(self) -> None:
        module = build_module("factorized_gated_lut", alpha=0.0)
        addresses = torch.arange(64, dtype=torch.long)
        response = torch.randn(64)
        output, _ = module(addresses, response)
        self.assertTrue(torch.equal(output, response))

    def test_existing_modes_still_produce_finite_outputs(self) -> None:
        modes = (
            "address_lut",
            "global_mean",
            "token_channel_lut",
            "shuffled_address_lut",
            "global_plus_address_lut",
            "global_plus_shuffled_address_lut",
        )
        addresses = torch.arange(64, dtype=torch.long)
        response = torch.randn(64)
        for mode in modes:
            output, index = build_module(mode)(addresses, response)
            self.assertEqual(output.shape, response.shape)
            self.assertEqual(index.shape, addresses.shape)
            self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
