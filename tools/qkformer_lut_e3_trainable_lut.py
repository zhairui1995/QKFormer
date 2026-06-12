#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from contextlib import nullcontext
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from qkformer_lut_e2_replace import (
    ModulePrototype,
    PrototypeBank,
    accuracy,
    build_cifar10_model,
    build_loader,
    calibrate_prototypes,
    load_checkpoint_if_available,
    load_config,
    reset_model_state,
    set_seed,
    spiking_self_full_address,
    token_qk_full_address,
)


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class TrainableLUTModule(nn.Module):
    def __init__(
        self,
        proto: ModulePrototype,
        mode: str,
        alpha_init: float,
        learn_alpha: bool,
        shrinkage_tau: float,
        address_scale: float,
        mode_seed: int,
        token_bins: int,
        channel_bins: int,
        population_bins: int,
    ) -> None:
        super().__init__()
        self.name = proto.name
        self.kind = proto.kind
        self.stage = proto.stage
        self.mode = str(mode)
        self.address_space = int(proto.address_space)
        self.address_scale = float(address_scale)
        self.mode_seed = int(mode_seed)
        self.token_bins = int(token_bins)
        self.channel_bins = int(channel_bins)
        self.population_bins = int(population_bins)
        supported_modes = {
            "address_lut",
            "global_mean",
            "token_channel_lut",
            "shuffled_address_lut",
            "global_plus_address_lut",
            "global_plus_shuffled_address_lut",
            "factorized_sum_lut",
            "factorized_gated_lut",
            "matched_param_address_lut",
            "factorized_shuffled_lut",
        }
        if self.mode not in supported_modes:
            raise ValueError(f"unsupported E3 adapter mode: {self.mode}")

        self.component_names, self.component_sizes = self._component_spec()
        self.factorized_modes = {
            "factorized_sum_lut",
            "factorized_gated_lut",
            "factorized_shuffled_lut",
        }
        if self.mode in self.factorized_modes:
            self.global_table = nn.Parameter(torch.tensor([float(proto.global_mean)], dtype=torch.float32))
            component_init = self._component_init(proto, shrinkage_tau)
            self.component_tables = nn.ParameterDict(
                {name: nn.Parameter(component_init[name]) for name in self.component_names}
            )
            if self.mode == "factorized_gated_lut":
                self.component_gate_logits = nn.Parameter(torch.zeros(len(self.component_names), dtype=torch.float32))
        elif self.mode == "matched_param_address_lut":
            matched_entries = 1 + sum(self.component_sizes) + len(self.component_names)
            self.table = nn.Parameter(self._hashed_address_init(proto, matched_entries, shrinkage_tau))
        elif self.mode == "global_mean":
            init = torch.tensor([float(proto.global_mean)], dtype=torch.float32)
        elif self.mode == "token_channel_lut":
            init = self._coarse_init(proto, shrinkage_tau)
        elif self.mode in {"global_plus_address_lut", "global_plus_shuffled_address_lut"}:
            address_init = self._address_init(proto, shrinkage_tau)
            init = address_init - address_init.mean()
            self.global_table = nn.Parameter(torch.tensor([float(proto.global_mean)], dtype=torch.float32))
        else:
            init = self._address_init(proto, shrinkage_tau)
        if self.mode not in self.factorized_modes and self.mode != "matched_param_address_lut":
            self.table = nn.Parameter(init)

        alpha_value = float(alpha_init)
        if learn_alpha:
            alpha = min(max(alpha_value, 1e-4), 1.0 - 1e-4)
            alpha_logit = math.log(alpha / (1.0 - alpha))
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
        else:
            self.register_buffer("fixed_alpha", torch.tensor(min(max(alpha_value, 0.0), 1.0), dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        if hasattr(self, "alpha_logit"):
            return torch.sigmoid(self.alpha_logit)
        return self.fixed_alpha

    def forward(self, address: torch.Tensor, response: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode in self.factorized_modes:
            pred, index = self._factorized_prediction(address, response)
            return response + self.alpha.to(device=response.device, dtype=response.dtype) * (pred - response), index
        index = self._index(address, response.device)
        table = self.table.to(device=response.device, dtype=response.dtype)
        if self.mode in {"global_plus_address_lut", "global_plus_shuffled_address_lut"}:
            centered_table = table - table.mean()
            global_value = self.global_table.to(device=response.device, dtype=response.dtype)[0]
            pred = global_value + self.address_scale * centered_table[index]
        else:
            pred = table[index]
        return response + self.alpha.to(device=response.device, dtype=response.dtype) * (pred - response), index

    def _index(self, address: torch.Tensor, device: torch.device) -> torch.Tensor:
        addr = address.detach().to(device=device, dtype=torch.long)
        if self.mode == "global_mean":
            return torch.zeros_like(addr)
        if self.mode == "token_channel_lut":
            detail_factor = 4 if self.kind == "token_qk" else 4 * self.population_bins * self.population_bins
            return torch.div(addr, detail_factor, rounding_mode="floor").clamp_max(self.table.numel() - 1)
        if self.mode == "matched_param_address_lut":
            return torch.remainder(addr, self.table.numel())
        if self.mode in {"shuffled_address_lut", "global_plus_shuffled_address_lut"}:
            return self._address_permutation(device)[addr]
        return addr

    def _address_permutation(self, device: torch.device) -> torch.Tensor:
        perm = getattr(self, "_cached_address_permutation", None)
        if perm is None:
            digest = hashlib.sha256(f"{self.mode_seed}:{self.name}".encode("utf-8")).hexdigest()
            seed = int(digest[:16], 16) % (2**63)
            generator = torch.Generator()
            generator.manual_seed(seed)
            perm = torch.randperm(self.address_space, generator=generator, dtype=torch.long)
            self._cached_address_permutation = perm
        return perm.to(device=device)

    def _component_spec(self) -> Tuple[List[str], List[int]]:
        detail = 4 if self.kind == "token_qk" else 4 * self.population_bins * self.population_bins
        denominator = self.token_bins * self.channel_bins * detail
        if denominator <= 0 or self.address_space % denominator != 0:
            raise ValueError(
                f"cannot factor address space for {self.name}: address_space={self.address_space}, denominator={denominator}"
            )
        heads = self.address_space // denominator
        names = ["head", "token", "channel", "q_bit", "k_bit"]
        sizes = [heads, self.token_bins, self.channel_bins, 2, 2]
        if self.kind == "spiking_self":
            names.extend(["q_population", "k_population"])
            sizes.extend([self.population_bins, self.population_bins])
        return names, sizes

    def split_address(self, address: torch.Tensor) -> Dict[str, torch.Tensor]:
        value = address.detach().to(dtype=torch.long)
        decoded: Dict[str, torch.Tensor] = {}
        for name, size in reversed(list(zip(self.component_names, self.component_sizes))):
            decoded[name] = torch.remainder(value, size)
            value = torch.div(value, size, rounding_mode="floor")
        return {name: decoded[name] for name in self.component_names}

    def _component_permutation(self, name: str, size: int, device: torch.device) -> torch.Tensor:
        cache_name = f"_cached_component_permutation_{name}"
        perm = getattr(self, cache_name, None)
        if perm is None:
            digest = hashlib.sha256(f"{self.mode_seed}:{self.name}:{name}".encode("utf-8")).hexdigest()
            generator = torch.Generator()
            generator.manual_seed(int(digest[:16], 16) % (2**63))
            perm = torch.randperm(size, generator=generator, dtype=torch.long)
            setattr(self, cache_name, perm)
        return perm.to(device=device)

    def _factorized_prediction(
        self, address: torch.Tensor, response: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        components = self.split_address(address.to(device=response.device))
        packed_index = torch.zeros_like(address, device=response.device, dtype=torch.long)
        residual = torch.zeros_like(response)
        if self.mode == "factorized_gated_lut":
            weights = torch.softmax(self.component_gate_logits, dim=0).to(response.dtype) * len(self.component_names)
        else:
            weights = None
        for component_idx, (name, size) in enumerate(zip(self.component_names, self.component_sizes)):
            index = components[name].to(device=response.device)
            if self.mode == "factorized_shuffled_lut":
                index = self._component_permutation(name, size, response.device)[index]
            table = self.component_tables[name].to(device=response.device, dtype=response.dtype)
            term = table[index]
            if weights is not None:
                term = term * weights[component_idx]
            residual = residual + term
            packed_index = packed_index * size + index
        global_value = self.global_table.to(device=response.device, dtype=response.dtype)[0]
        return global_value + residual, packed_index

    def _address_init(self, proto: ModulePrototype, shrinkage_tau: float) -> torch.Tensor:
        means = proto.address_mean.to(torch.float32)
        counts = proto.count.to(torch.float32)
        global_mean = torch.full_like(means, fill_value=float(proto.global_mean))
        if shrinkage_tau <= 0:
            return means
        weight = counts / (counts + float(shrinkage_tau))
        return weight * means + (1.0 - weight) * global_mean

    def _coarse_init(self, proto: ModulePrototype, shrinkage_tau: float) -> torch.Tensor:
        detail_factor = 4 if self.kind == "token_qk" else 4 * self.population_bins * self.population_bins
        coarse_space = max(1, proto.address_space // detail_factor)
        coarse_sum = torch.zeros(coarse_space, dtype=torch.float64)
        coarse_count = torch.zeros(coarse_space, dtype=torch.float64)
        coarse_index = torch.arange(proto.address_space, dtype=torch.long) // detail_factor
        coarse_sum.scatter_add_(0, coarse_index, proto.sum)
        coarse_count.scatter_add_(0, coarse_index, proto.count)
        global_mean = torch.full((coarse_space,), fill_value=float(proto.global_mean), dtype=torch.float64)
        seen = coarse_count > 0
        means = global_mean.clone()
        means[seen] = coarse_sum[seen] / coarse_count[seen]
        if shrinkage_tau > 0:
            weight = coarse_count / (coarse_count + float(shrinkage_tau))
            means = weight * means + (1.0 - weight) * global_mean
        return means.to(torch.float32)

    def _component_init(self, proto: ModulePrototype, shrinkage_tau: float) -> Dict[str, torch.Tensor]:
        all_addresses = torch.arange(self.address_space, dtype=torch.long)
        components = self.split_address(all_addresses)
        output: Dict[str, torch.Tensor] = {}
        for name, size in zip(self.component_names, self.component_sizes):
            sums = torch.zeros(size, dtype=torch.float64)
            counts = torch.zeros(size, dtype=torch.float64)
            sums.scatter_add_(0, components[name], proto.sum)
            counts.scatter_add_(0, components[name], proto.count)
            means = torch.full((size,), float(proto.global_mean), dtype=torch.float64)
            seen = counts > 0
            means[seen] = sums[seen] / counts[seen]
            if shrinkage_tau > 0:
                weight = counts / (counts + float(shrinkage_tau))
                means = weight * means + (1.0 - weight) * float(proto.global_mean)
            output[name] = (means - float(proto.global_mean)).to(torch.float32)
        return output

    def _hashed_address_init(
        self, proto: ModulePrototype, table_entries: int, shrinkage_tau: float
    ) -> torch.Tensor:
        index = torch.remainder(torch.arange(self.address_space, dtype=torch.long), table_entries)
        sums = torch.zeros(table_entries, dtype=torch.float64)
        counts = torch.zeros(table_entries, dtype=torch.float64)
        sums.scatter_add_(0, index, proto.sum)
        counts.scatter_add_(0, index, proto.count)
        means = torch.full((table_entries,), float(proto.global_mean), dtype=torch.float64)
        seen = counts > 0
        means[seen] = sums[seen] / counts[seen]
        if shrinkage_tau > 0:
            weight = counts / (counts + float(shrinkage_tau))
            means = weight * means + (1.0 - weight) * float(proto.global_mean)
        return means.to(torch.float32)

    def summary(self) -> Dict[str, object]:
        if self.mode in self.factorized_modes:
            table_entries = 1 + sum(table.numel() for table in self.component_tables.values())
            active_entries = table_entries
            component_entries = {
                name: int(self.component_tables[name].numel()) for name in self.component_names
            }
        else:
            table_entries = int(self.table.numel())
            active_entries = int(torch.count_nonzero(self.table.detach()).item())
            component_entries = None
        summary = {
            "name": self.name,
            "kind": self.kind,
            "stage": self.stage,
            "mode": self.mode,
            "address_space": self.address_space,
            "table_entries": table_entries,
            "active_entries": active_entries,
            "estimated_table_bytes_fp32": 4 * table_entries,
            "estimated_trainable_bytes_fp32": 4
            * sum(param.numel() for param in self.parameters() if param.requires_grad),
            "trainable_parameters": sum(param.numel() for param in self.parameters() if param.requires_grad),
            "alpha": float(self.alpha.detach().cpu().item()),
        }
        if component_entries is not None:
            summary["component_entries"] = component_entries
            summary["component_order"] = list(self.component_names)
            summary["composition"] = "gated_sum" if self.mode == "factorized_gated_lut" else "sum"
            if hasattr(self, "component_gate_logits"):
                summary["component_weights"] = {
                    name: float(weight)
                    for name, weight in zip(
                        self.component_names,
                        (torch.softmax(self.component_gate_logits.detach(), dim=0) * len(self.component_names)).cpu(),
                    )
                }
        if self.mode in {"global_plus_address_lut", "global_plus_shuffled_address_lut"}:
            summary["global_value"] = float(self.global_table.detach().cpu().item())
            summary["centered_table_mean"] = float((self.table - self.table.mean()).detach().mean().cpu().item())
            summary["address_scale"] = self.address_scale
        return summary


class TrainableLUTAdapter(nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        prototypes: Dict[str, ModulePrototype],
        target_modules: Iterable[str],
        token_bins: int,
        channel_bins: int,
        population_bins: int,
        mode: str,
        alpha_init: float,
        learn_alpha: bool,
        shrinkage_tau: float,
        address_scale: float,
        mode_seed: int,
    ) -> None:
        super().__init__()
        # Keep the frozen backbone outside this module tree. Registering it as a
        # child would let adapter.train() silently switch BatchNorm layers back
        # to training mode and mutate the baseline running statistics.
        self.targets = set(target_modules)
        self.token_bins = int(token_bins)
        self.channel_bins = int(channel_bins)
        self.population_bins = int(population_bins)
        self.mode = str(mode)
        self.modules_by_name = dict(model.named_modules())
        self.luts = nn.ModuleDict()
        for name in self.targets:
            if name in prototypes:
                self.luts[name.replace(".", "__")] = TrainableLUTModule(
                    prototypes[name],
                    mode=self.mode,
                    alpha_init=alpha_init,
                    learn_alpha=learn_alpha,
                    shrinkage_tau=shrinkage_tau,
                    address_scale=address_scale,
                    mode_seed=mode_seed,
                    token_bins=self.token_bins,
                    channel_bins=self.channel_bins,
                    population_bins=self.population_bins,
                )
        self.enabled = False
        self.buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.local_losses: List[torch.Tensor] = []
        self.replaced_values = 0
        self._register()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear_step(self) -> None:
        self.local_losses.clear()
        self.replaced_values = 0

    def local_mse_loss(self, device: torch.device) -> torch.Tensor:
        if not self.local_losses:
            return torch.zeros((), device=device)
        return torch.stack(self.local_losses).mean()

    def _register(self) -> None:
        for name, module in self.modules_by_name.items():
            if name not in self.targets or name.replace(".", "__") not in self.luts:
                continue
            for child_name in ("q_lif", "k_lif", "attn_lif", "proj_lif"):
                child = getattr(module, child_name, None)
                if child is not None:
                    self.handles.append(child.register_forward_hook(self._make_hook(name, child_name.replace("_lif", ""))))

    def _make_hook(self, prefix: str, kind: str):
        def hook(_module, _inputs, output):
            self.buffers[prefix][kind] = output.detach()
            if kind != "proj":
                return None
            try:
                if not self.enabled:
                    return None
                return self._replace(prefix, output)
            finally:
                self.buffers[prefix].clear()

        return hook

    def _replace(self, prefix: str, output: torch.Tensor) -> Optional[torch.Tensor]:
        buf = self.buffers[prefix]
        if "q" not in buf or "k" not in buf:
            return None
        module = self.modules_by_name[prefix]
        cls_name = module.__class__.__name__
        if cls_name == "Token_QK_Attention" and "attn" in buf:
            address, response, layout, _kind = token_qk_full_address(
                buf["q"], buf["k"], buf["attn"], output, self.token_bins, self.channel_bins
            )
            replacement_flat = self._predict(prefix, address, response.to(output.device, output.dtype))
            replacement = replacement_flat.reshape(layout).reshape_as(output)
        elif cls_name == "Spiking_Self_Attention":
            address, response, layout, _kind = spiking_self_full_address(
                module,
                buf["q"],
                buf["k"],
                output,
                self.token_bins,
                self.channel_bins,
                self.population_bins,
            )
            replacement_flat = self._predict(prefix, address, response.to(output.device, output.dtype))
            t, b, c, n = layout
            heads = int(getattr(module, "num_heads", 1))
            depth = c // heads
            replacement = replacement_flat.reshape(t, b, heads, n, depth).permute(0, 1, 2, 4, 3).reshape(t, b, c, n)
            replacement = replacement.reshape_as(output)
        else:
            return None
        self.local_losses.append(F.mse_loss(replacement.float(), output.detach().float()))
        self.replaced_values += int(output.numel())
        return replacement

    def _predict(self, prefix: str, address: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        lut = self.luts[prefix.replace(".", "__")]
        pred, _index = lut(address, response)
        return pred

    def summary(self) -> Dict[str, object]:
        module_summaries = {name.replace("__", "."): module.summary() for name, module in self.luts.items()}
        return {
            "mode": self.mode,
            "target_modules": sorted(self.targets),
            "num_modules": len(self.luts),
            "trainable_parameters": sum(param.numel() for param in self.parameters() if param.requires_grad),
            "module_summary": module_summaries,
        }


def apply_env_overrides(model_cfg, data_cfg, adapter_cfg, train_cfg):
    def env_bool(name: str) -> Optional[bool]:
        value = os.environ.get(name)
        if value is None:
            return None
        return value.strip().lower() in {"1", "true", "yes", "on"}

    env_checkpoint = os.environ.get("QKFORMER_LUT_CKPT")
    if env_checkpoint:
        model_cfg["checkpoint"] = env_checkpoint
    env_time_step = os.environ.get("QKFORMER_LUT_TIME_STEP")
    if env_time_step:
        model_cfg["time_step"] = int(env_time_step)
    env_data_dir = os.environ.get("QKFORMER_LUT_DATA_DIR")
    if env_data_dir:
        for split_cfg in data_cfg.values():
            split_cfg["data_dir"] = env_data_dir
    env_seed = os.environ.get("QKFORMER_LUT_E3_SEED")
    if env_seed:
        data_cfg["calibration"]["seed"] = int(env_seed)
        data_cfg["train"]["seed"] = int(env_seed) + 1
        adapter_cfg["mode_seed"] = int(env_seed)
    env_mode = os.environ.get("QKFORMER_LUT_E3_MODE")
    if env_mode:
        adapter_cfg["mode"] = env_mode
    env_targets = os.environ.get("QKFORMER_LUT_E3_TARGETS")
    if env_targets:
        adapter_cfg["target_modules"] = [item.strip() for item in env_targets.split(",") if item.strip()]
    env_epochs = os.environ.get("QKFORMER_LUT_E3_EPOCHS")
    if env_epochs:
        train_cfg["epochs"] = int(env_epochs)
    env_lr = os.environ.get("QKFORMER_LUT_E3_LR")
    if env_lr:
        train_cfg["lr"] = float(env_lr)
    env_lambda_ce = os.environ.get("QKFORMER_LUT_E3_LAMBDA_CE")
    if env_lambda_ce:
        train_cfg["lambda_ce"] = float(env_lambda_ce)
    env_lambda_kl = os.environ.get("QKFORMER_LUT_E3_LAMBDA_KL")
    if env_lambda_kl:
        train_cfg["lambda_kl"] = float(env_lambda_kl)
    env_lambda_local = os.environ.get("QKFORMER_LUT_E3_LAMBDA_LOCAL_MSE")
    if env_lambda_local:
        train_cfg["lambda_local_mse"] = float(env_lambda_local)
    env_alpha_init = os.environ.get("QKFORMER_LUT_E3_ALPHA_INIT")
    if env_alpha_init:
        adapter_cfg["alpha_init"] = float(env_alpha_init)
    env_learn_alpha = env_bool("QKFORMER_LUT_E3_LEARN_ALPHA")
    if env_learn_alpha is not None:
        adapter_cfg["learn_alpha"] = env_learn_alpha
    env_shrinkage_tau = os.environ.get("QKFORMER_LUT_E3_SHRINKAGE_TAU")
    if env_shrinkage_tau:
        adapter_cfg["shrinkage_tau"] = float(env_shrinkage_tau)
    env_address_scale = os.environ.get("QKFORMER_LUT_E3_ADDRESS_SCALE")
    if env_address_scale:
        adapter_cfg["address_scale"] = float(env_address_scale)
    env_calib_batches = os.environ.get("QKFORMER_LUT_E3_CALIB_BATCHES")
    if env_calib_batches:
        data_cfg["calibration"]["num_batches"] = int(env_calib_batches)
    env_train_batches = os.environ.get("QKFORMER_LUT_E3_TRAIN_BATCHES")
    if env_train_batches:
        data_cfg["train"]["num_batches"] = int(env_train_batches)
    env_eval_batches = os.environ.get("QKFORMER_LUT_E3_EVAL_BATCHES")
    if env_eval_batches:
        data_cfg["evaluation"]["num_batches"] = int(env_eval_batches)
    env_eval_batch_size = os.environ.get("QKFORMER_LUT_E3_EVAL_BATCH_SIZE")
    if env_eval_batch_size:
        data_cfg["evaluation"]["batch_size"] = int(env_eval_batch_size)
    env_eval_amp = env_bool("QKFORMER_LUT_E3_EVAL_AMP")
    if env_eval_amp is not None:
        data_cfg["evaluation"]["amp"] = env_eval_amp
    env_eval_loader = os.environ.get("QKFORMER_LUT_E3_EVAL_LOADER")
    if env_eval_loader:
        data_cfg["evaluation"]["backend"] = env_eval_loader
    env_save_per_sample = env_bool("QKFORMER_LUT_E3_SAVE_PER_SAMPLE")
    if env_save_per_sample is not None:
        data_cfg["evaluation"]["save_per_sample"] = env_save_per_sample
    return {
        "QKFORMER_LUT_CKPT": env_checkpoint,
        "QKFORMER_LUT_TIME_STEP": env_time_step,
        "QKFORMER_LUT_DATA_DIR": env_data_dir,
        "QKFORMER_LUT_E3_SEED": env_seed,
        "QKFORMER_LUT_E3_MODE": env_mode,
        "QKFORMER_LUT_E3_TARGETS": env_targets,
        "QKFORMER_LUT_E3_EPOCHS": env_epochs,
        "QKFORMER_LUT_E3_LR": env_lr,
        "QKFORMER_LUT_E3_LAMBDA_CE": env_lambda_ce,
        "QKFORMER_LUT_E3_LAMBDA_KL": env_lambda_kl,
        "QKFORMER_LUT_E3_LAMBDA_LOCAL_MSE": env_lambda_local,
        "QKFORMER_LUT_E3_ALPHA_INIT": env_alpha_init,
        "QKFORMER_LUT_E3_LEARN_ALPHA": os.environ.get("QKFORMER_LUT_E3_LEARN_ALPHA"),
        "QKFORMER_LUT_E3_SHRINKAGE_TAU": env_shrinkage_tau,
        "QKFORMER_LUT_E3_ADDRESS_SCALE": env_address_scale,
        "QKFORMER_LUT_E3_CALIB_BATCHES": env_calib_batches,
        "QKFORMER_LUT_E3_TRAIN_BATCHES": env_train_batches,
        "QKFORMER_LUT_E3_EVAL_BATCHES": env_eval_batches,
        "QKFORMER_LUT_E3_EVAL_BATCH_SIZE": env_eval_batch_size,
        "QKFORMER_LUT_E3_EVAL_AMP": os.environ.get("QKFORMER_LUT_E3_EVAL_AMP"),
        "QKFORMER_LUT_E3_EVAL_LOADER": env_eval_loader,
        "QKFORMER_LUT_E3_SAVE_PER_SAMPLE": os.environ.get("QKFORMER_LUT_E3_SAVE_PER_SAMPLE"),
    }


def train_one_epoch(model, adapter, loader, data_cfg, train_cfg, optimizer, device) -> Dict[str, float]:
    model.eval()
    adapter.train()
    if model.training:
        raise RuntimeError("adapter.train() must not switch the frozen backbone to training mode")
    loss_m = AverageMeter()
    ce_m = AverageMeter()
    kl_m = AverageMeter()
    local_m = AverageMeter()
    top1_m = AverageMeter()
    max_batches = int(data_cfg["num_batches"])
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = int(targets.numel())

        adapter.enabled = False
        reset_model_state(model)
        with torch.no_grad():
            teacher_logits = model(images)
        reset_model_state(model)

        adapter.clear_step()
        adapter.enabled = True
        reset_model_state(model)
        student_logits = model(images)
        ce = F.cross_entropy(student_logits, targets)
        kl = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1), reduction="batchmean")
        local = adapter.local_mse_loss(device)
        loss = (
            float(train_cfg.get("lambda_ce", 1.0)) * ce
            + float(train_cfg.get("lambda_kl", 0.5)) * kl
            + float(train_cfg.get("lambda_local_mse", 0.05)) * local
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), clip_norm)
        optimizer.step()
        reset_model_state(model)
        adapter.enabled = False

        top1 = accuracy(student_logits.detach(), targets, topk=(1,))[0]
        loss_m.update(float(loss.item()), batch_size)
        ce_m.update(float(ce.item()), batch_size)
        kl_m.update(float(kl.item()), batch_size)
        local_m.update(float(local.item()), batch_size)
        top1_m.update(top1, batch_size)
    return {"loss": loss_m.mean, "ce": ce_m.mean, "kl": kl_m.mean, "local_mse": local_m.mean, "top1": top1_m.mean}


def _true_class_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets.view(-1, 1), float("-inf"))
    max_other = masked.max(dim=1).values
    return target_logits - max_other


def _append_per_sample_rows(
    rows: List[Dict[str, object]],
    sample_offset: int,
    targets: torch.Tensor,
    baseline_logits: torch.Tensor,
    replacement_logits: torch.Tensor,
) -> int:
    base_logits = baseline_logits.detach().float()
    repl_logits = replacement_logits.detach().float()
    labels = targets.detach().long()
    base_ce = F.cross_entropy(base_logits, labels, reduction="none")
    repl_ce = F.cross_entropy(repl_logits, labels, reduction="none")
    base_pred = base_logits.argmax(dim=1)
    repl_pred = repl_logits.argmax(dim=1)
    base_margin = _true_class_margin(base_logits, labels)
    repl_margin = _true_class_margin(repl_logits, labels)
    base_conf = F.softmax(base_logits, dim=1).max(dim=1).values
    repl_conf = F.softmax(repl_logits, dim=1).max(dim=1).values

    labels_cpu = labels.cpu()
    for idx in range(int(labels.numel())):
        rows.append(
            {
                "sample_index": sample_offset + idx,
                "target": int(labels_cpu[idx].item()),
                "baseline_top1": int(base_pred[idx].cpu().item()),
                "replacement_top1": int(repl_pred[idx].cpu().item()),
                "baseline_correct": int(base_pred[idx].cpu().item() == labels_cpu[idx].item()),
                "replacement_correct": int(repl_pred[idx].cpu().item() == labels_cpu[idx].item()),
                "baseline_ce": float(base_ce[idx].cpu().item()),
                "replacement_ce": float(repl_ce[idx].cpu().item()),
                "baseline_margin": float(base_margin[idx].cpu().item()),
                "replacement_margin": float(repl_margin[idx].cpu().item()),
                "baseline_confidence": float(base_conf[idx].cpu().item()),
                "replacement_confidence": float(repl_conf[idx].cpu().item()),
            }
        )
    return sample_offset + int(labels.numel())


def _write_per_sample_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "target",
        "baseline_top1",
        "replacement_top1",
        "baseline_correct",
        "replacement_correct",
        "baseline_ce",
        "replacement_ce",
        "baseline_margin",
        "replacement_margin",
        "baseline_confidence",
        "replacement_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, adapter, loader, data_cfg, device, per_sample_path: Optional[Path] = None) -> Dict[str, object]:
    model.eval()
    adapter.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)
    base_loss = AverageMeter()
    repl_loss = AverageMeter()
    base_top1 = AverageMeter()
    repl_top1 = AverageMeter()
    base_top5 = AverageMeter()
    repl_top5 = AverageMeter()
    logit_mse = AverageMeter()
    kl_m = AverageMeter()
    local_m = AverageMeter()
    max_batches = int(data_cfg["num_batches"])
    per_sample_rows: List[Dict[str, object]] = []
    sample_offset = 0
    use_amp = bool(data_cfg.get("amp", False)) and device.type == "cuda"
    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = int(targets.numel())
            adapter.enabled = False
            reset_model_state(model)
            with amp_context():
                baseline_logits = model(images)
                b_loss = loss_fn(baseline_logits, targets)
            b_top1, b_top5 = accuracy(baseline_logits, targets)
            reset_model_state(model)

            adapter.clear_step()
            adapter.enabled = True
            reset_model_state(model)
            with amp_context():
                replacement_logits = model(images)
                r_loss = loss_fn(replacement_logits, targets)
            r_top1, r_top5 = accuracy(replacement_logits, targets)
            local = adapter.local_mse_loss(device)
            kl = F.kl_div(F.log_softmax(replacement_logits, dim=1), F.softmax(baseline_logits, dim=1), reduction="batchmean")
            reset_model_state(model)

            base_loss.update(float(b_loss.item()), batch_size)
            repl_loss.update(float(r_loss.item()), batch_size)
            base_top1.update(b_top1, batch_size)
            repl_top1.update(r_top1, batch_size)
            base_top5.update(b_top5, batch_size)
            repl_top5.update(r_top5, batch_size)
            logit_mse.update(float(torch.mean((replacement_logits - baseline_logits) ** 2).item()), batch_size)
            kl_m.update(float(kl.item()), batch_size)
            local_m.update(float(local.item()), batch_size)
            if per_sample_path is not None:
                sample_offset = _append_per_sample_rows(
                    per_sample_rows,
                    sample_offset,
                    targets,
                    baseline_logits,
                    replacement_logits,
                )
    adapter.enabled = False
    result = {
        "baseline": {"loss": base_loss.mean, "top1": base_top1.mean, "top5": base_top5.mean},
        "replacement": {
            "loss": repl_loss.mean,
            "top1": repl_top1.mean,
            "top5": repl_top5.mean,
            "logit_mse": logit_mse.mean,
            "kl_to_baseline": kl_m.mean,
            "local_mse": local_m.mean,
        },
        "delta": {
            "loss": repl_loss.mean - base_loss.mean,
            "top1": repl_top1.mean - base_top1.mean,
            "top5": repl_top5.mean - base_top5.mean,
        },
    }
    if per_sample_path is not None:
        _write_per_sample_csv(per_sample_path, per_sample_rows)
        result["per_sample"] = {
            "path": str(per_sample_path),
            "num_samples": len(per_sample_rows),
            "schema": "sample_index,target,baseline_top1,replacement_top1,baseline_correct,replacement_correct,baseline_ce,replacement_ce,baseline_margin,replacement_margin,baseline_confidence,replacement_confidence",
        }
    return result


def run(config_path: Path, output_dir: Path) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(config_path)
    model_cfg = dict(cfg["model"])
    data_cfg = {key: dict(value) for key, value in dict(cfg["data"]).items()}
    diag_cfg = dict(cfg["diagnostic"])
    adapter_cfg = dict(cfg["adapter"])
    train_cfg = dict(cfg["train"])
    env_overrides = apply_env_overrides(model_cfg, data_cfg, adapter_cfg, train_cfg)
    run_seed = int(os.environ.get("QKFORMER_LUT_E3_SEED", cfg["experiment"].get("seed", 42)))
    set_seed(run_seed)

    device_name = str(diag_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the upstream cupy-backed QKFormer LIF modules")
    device = torch.device(device_name)
    model = build_cifar10_model(root, model_cfg)
    checkpoint_info = load_checkpoint_if_available(model, model_cfg.get("checkpoint"))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    calibration_loader = build_loader(data_cfg["calibration"], device)
    train_loader = build_loader(data_cfg["train"], device)
    evaluation_loader = build_loader(data_cfg["evaluation"], device)
    bank = PrototypeBank(min_count=int(diag_cfg.get("prototype_min_count", 2)))
    print("[qk-lut-e3] calibration_start")
    calibration_batches, calibration_hook_summary = calibrate_prototypes(
        model, calibration_loader, data_cfg["calibration"], diag_cfg, bank, device
    )
    print(f"[qk-lut-e3] calibration_batches={calibration_batches}")

    target_modules = list(adapter_cfg.get("target_modules", [])) or ["stage1.0.tssa"]
    adapter = TrainableLUTAdapter(
        model,
        bank.prototypes,
        target_modules,
        token_bins=int(diag_cfg.get("token_bins", 8)),
        channel_bins=int(diag_cfg.get("channel_bins", 8)),
        population_bins=int(diag_cfg.get("population_bins", 4)),
        mode=str(adapter_cfg.get("mode", "address_lut")),
        alpha_init=float(adapter_cfg.get("alpha_init", 0.25)),
        learn_alpha=bool(adapter_cfg.get("learn_alpha", True)),
        shrinkage_tau=float(adapter_cfg.get("shrinkage_tau", 256.0)),
        address_scale=float(adapter_cfg.get("address_scale", 1.0)),
        mode_seed=int(adapter_cfg.get("mode_seed", 0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(train_cfg.get("lr", 0.01)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    train_history = []
    print(f"[qk-lut-e3] adapter_mode={adapter_cfg.get('mode')}")
    print(f"[qk-lut-e3] trainable_parameters={adapter.summary()['trainable_parameters']}")
    for epoch in range(int(train_cfg.get("epochs", 1))):
        metrics = train_one_epoch(model, adapter, train_loader, data_cfg["train"], train_cfg, optimizer, device)
        train_history.append({"epoch": epoch, **metrics})
        print(
            f"[qk-lut-e3] epoch={epoch} loss={metrics['loss']:.6f} "
            f"ce={metrics['ce']:.6f} kl={metrics['kl']:.6f} "
            f"local_mse={metrics['local_mse']:.6f} top1={metrics['top1']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = None
    if bool(data_cfg["evaluation"].get("save_per_sample", False)):
        per_sample_path = output_dir / "per_sample_predictions.csv"
        print(f"[qk-lut-e3] per_sample_path={per_sample_path}")
    print("[qk-lut-e3] evaluation_start")
    evaluation = evaluate(model, adapter, evaluation_loader, data_cfg["evaluation"], device, per_sample_path)
    adapter.close()
    metrics = {
        "experiment": {**cfg["experiment"], "seed": run_seed},
        "model": {
            "family": model_cfg["family"],
            "time_step": model_cfg["time_step"],
            "layer": model_cfg["layer"],
            "dim": model_cfg["dim"],
            "num_heads": model_cfg["num_heads"],
            "checkpoint": checkpoint_info,
        },
        "data": {
            "calibration": {"source": "cifar10", **data_cfg["calibration"], "processed_batches": calibration_batches},
            "train": {"source": "cifar10", **data_cfg["train"]},
            "evaluation": {"source": "cifar10", **data_cfg["evaluation"]},
        },
        "diagnostic_config": diag_cfg,
        "adapter_config": adapter_cfg,
        "train_config": train_cfg,
        "protocol": {
            "version": 4,
            "frozen_backbone_kept_in_eval_mode": True,
            "evaluation_batch_size": int(data_cfg["evaluation"]["batch_size"]),
            "evaluation_amp": bool(data_cfg["evaluation"].get("amp", False)),
            "evaluation_loader": str(data_cfg["evaluation"].get("backend", "torchvision")),
        },
        "env_overrides": env_overrides,
        "adapter_summary": adapter.summary(),
        "calibration_prototypes": bank.summary(),
        "calibration_hook_summary": calibration_hook_summary,
        "train_history": train_history,
        "classification": evaluation,
        "verdict": "PENDING_PHASE_GATE_REVIEW" if checkpoint_info["loaded"] else "PENDING_REAL_DATA_CHECKPOINT",
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"[qk-lut-e3] wrote {metrics_path}")
    print(f"[qk-lut-e3] baseline_top1={evaluation['baseline']['top1']}")
    print(f"[qk-lut-e3] replacement_top1={evaluation['replacement']['top1']}")
    print(f"[qk-lut-e3] delta_top1={evaluation['delta']['top1']}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QK-LUTFormer E3 trainable residual LUT adapter")
    parser.add_argument("--config", type=Path, default=Path("configs/qkformer_lut_e3_trainable_lut.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
