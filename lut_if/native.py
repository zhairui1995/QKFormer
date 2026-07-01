from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Pattern, Tuple

import torch
from torch import nn

from .neuron import DenseLUTIFNeuron
from .replace import set_submodule


ATTENTION_MARKERS = (".tssa.", ".ssa.", ".attn.")
MLP_MARKERS = (".mlp.",)


@dataclass(frozen=True)
class NativeQKLUTLIFConfig:
    target_scope: str = "attention"
    name_regex: str = ""
    state_bits: int = 6
    input_bits: int = 8
    x_range: Tuple[float, float] = (-8.0, 8.0)
    v_range: Tuple[float, float] = (0.0, 2.0)
    surrogate_slope: float = 2.0
    learn_threshold: bool = True


def _is_lif_like(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ in {"MultiStepLIFNode", "MultiStepParametricLIFNode"}
        and hasattr(module, "tau")
        and hasattr(module, "v_threshold")
    )


def _compile_regex(pattern: str) -> Pattern[str] | None:
    return re.compile(pattern) if pattern else None


def _scope_match(name: str, scope: str, regex: Pattern[str] | None) -> bool:
    if regex is not None and regex.search(name) is None:
        return False
    qualified = f".{name}."
    if scope == "all":
        return True
    if scope == "attention":
        return any(marker in qualified for marker in ATTENTION_MARKERS)
    if scope == "mlp":
        return any(marker in qualified for marker in MLP_MARKERS)
    if scope == "attention_mlp":
        return any(marker in qualified for marker in ATTENTION_MARKERS + MLP_MARKERS)
    if scope == "qk":
        return any(marker in qualified for marker in ATTENTION_MARKERS) and name.endswith(
            ("q_lif", "k_lif", "attn_lif")
        )
    raise ValueError(f"unsupported QK-LUT-LIF native target scope: {scope}")


def discover_native_lif_targets(
    model: nn.Module,
    *,
    target_scope: str = "attention",
    name_regex: str = "",
) -> Dict[str, nn.Module]:
    regex = _compile_regex(name_regex)
    targets: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if not name or not _is_lif_like(module):
            continue
        if _scope_match(name, target_scope, regex):
            targets[name] = module
    return targets


def _common(module: nn.Module) -> Dict[str, object]:
    return {
        "tau": float(getattr(module, "tau")),
        "decay_input": bool(getattr(module, "decay_input", False)),
        "v_threshold": float(getattr(module, "v_threshold")),
        "v_reset": getattr(module, "v_reset", None),
    }


def _replacement_device_dtype(original: nn.Module, fallback: nn.Module) -> tuple[torch.device, torch.dtype]:
    for tensor in list(original.parameters(recurse=False)) + list(original.buffers(recurse=False)):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    for tensor in list(fallback.parameters()) + list(fallback.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.float32


def apply_native_qklut_lif(
    model: nn.Module,
    config: NativeQKLUTLIFConfig,
) -> Dict[str, DenseLUTIFNeuron]:
    targets = discover_native_lif_targets(
        model,
        target_scope=config.target_scope,
        name_regex=config.name_regex,
    )
    replacements: Dict[str, DenseLUTIFNeuron] = {}
    for name, original in list(targets.items()):
        replacement = DenseLUTIFNeuron(
            **_common(original),
            x_range=config.x_range,
            v_range=config.v_range,
            state_bits=config.state_bits,
            input_bits=config.input_bits,
            surrogate_slope=config.surrogate_slope,
            learn_threshold=config.learn_threshold,
            hard_eval=True,
        )
        device, dtype = _replacement_device_dtype(original, model)
        replacement = replacement.to(device=device, dtype=dtype)
        set_submodule(model, name, replacement)
        replacements[name] = replacement
    return replacements


def native_qklut_lif_summary(modules: Mapping[str, DenseLUTIFNeuron]) -> Dict[str, object]:
    entries = sum(int(module.table_entries()) for module in modules.values())
    return {
        "modules": len(modules),
        "table_entries": entries,
        "value_bytes_fp32": entries * 4,
        "value_kib_fp32": entries * 4.0 / 1024.0,
        "targets": sorted(modules.keys()),
    }


def iter_native_lut_modules(model: nn.Module) -> Iterable[DenseLUTIFNeuron]:
    for module in model.modules():
        if isinstance(module, DenseLUTIFNeuron):
            yield module
