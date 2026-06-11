#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import yaml

from qkformer_lut.hooks import ModuleDiagnostic, QKAddressDiagnostic
from qkformer_lut.stats import VarianceStats


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cifar10_model(root: Path, cfg: Dict[str, object]) -> torch.nn.Module:
    family = str(cfg.get("family", "cifar10"))
    if family not in {"cifar10", "cifar100"}:
        raise ValueError(f"unsupported QKFormer model family: {family}")
    model_dir = root / family
    sys.path.insert(0, str(model_dir))
    try:
        model_module = importlib.import_module("model")
        model = model_module.spiking_transformer(
            drop_rate=0.0,
            drop_path_rate=0.0,
            img_size_h=int(cfg["img_size"]),
            img_size_w=int(cfg["img_size"]),
            patch_size=int(cfg["patch_size"]),
            embed_dims=int(cfg["dim"]),
            num_heads=int(cfg["num_heads"]),
            mlp_ratios=int(cfg["mlp_ratio"]),
            in_channels=int(cfg["in_channels"]),
            num_classes=int(cfg["num_classes"]),
            qkv_bias=False,
            depths=int(cfg["layer"]),
            sr_ratios=1,
            T=int(cfg["time_step"]),
        )
    finally:
        try:
            sys.path.remove(str(model_dir))
        except ValueError:
            pass
    return model


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint: Optional[str]) -> Dict[str, object]:
    if not checkpoint:
        return {"loaded": False, "path": None, "missing_keys": None, "unexpected_keys": None}
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get("state_dict") or payload.get("model") or payload
    else:
        state = payload
    state = _strip_module_prefix(state)
    msg = model.load_state_dict(state, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }


def make_synthetic_loader(
    batch_size: int,
    batches: int,
    in_channels: int,
    img_size: int,
    device: torch.device,
) -> Iterable[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    for _ in range(batches):
        images = torch.randn(batch_size, in_channels, img_size, img_size, device=device)
        yield images, None


def make_cifar10_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool = False,
    seed: int = 42,
) -> Tuple[Iterable[Tuple[torch.Tensor, Optional[torch.Tensor]]], str]:
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    root = Path(data_dir).expanduser()
    train = split == "train"
    dataset = datasets.CIFAR10(
        root=str(root),
        train=train,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, "cifar10"


def make_cifar100_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool = False,
    seed: int = 42,
) -> Tuple[Iterable[Tuple[torch.Tensor, Optional[torch.Tensor]]], str]:
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    root = Path(data_dir).expanduser()
    dataset = datasets.CIFAR100(
        root=str(root),
        train=split == "train",
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, "cifar100"


def build_loader(cfg: Dict[str, object], model_cfg: Dict[str, object], device: torch.device):
    mode = str(cfg.get("mode", "auto"))
    batch_size = int(cfg["batch_size"])
    family = str(model_cfg.get("family", "cifar10"))
    selected_mode = family if mode == "auto" else mode
    loaders = {
        "cifar10": make_cifar10_loader,
        "cifar100": make_cifar100_loader,
    }
    if selected_mode in loaders:
        try:
            return loaders[selected_mode](
                data_dir=str(cfg["data_dir"]),
                split=str(cfg.get("split", "validation")),
                batch_size=batch_size,
                workers=int(cfg.get("workers", 4)),
                device=device,
                shuffle=bool(cfg.get("shuffle", False)),
                seed=int(cfg.get("seed", 42)),
            )
        except Exception as exc:
            if mode != "auto":
                raise
            print(f"[qk-lut-e1] {selected_mode} loader unavailable, using synthetic input: {exc}")
    batches = int(cfg.get("synthetic_batches", cfg.get("num_batches", 4)))
    return (
        make_synthetic_loader(
            batch_size=batch_size,
            batches=batches,
            in_channels=int(model_cfg["in_channels"]),
            img_size=int(model_cfg["img_size"]),
            device=device,
        ),
        "synthetic",
    )


@dataclass
class MSEStats:
    count: int = 0
    sse: float = 0.0

    def update(self, squared_error_sum: float, count: int) -> None:
        self.count += int(count)
        self.sse += float(squared_error_sum)

    @property
    def mse(self) -> float:
        return self.sse / self.count if self.count > 0 else 0.0


class ModulePrototype:
    def __init__(self, stats: ModuleDiagnostic, min_count: int, population_bins: int, seed: int) -> None:
        self.name = stats.name
        self.kind = stats.kind
        self.stage = stats.stage
        self.block = stats.block
        self.address_space = int(stats.address_space)
        self.min_count = int(min_count)
        self.seed = int(seed)
        self.detail_factor = 4 if stats.kind == "token_qk" else 4 * int(population_bins) ** 2
        self.coarse_address_space = self.address_space // self.detail_factor
        self.count = torch.zeros(self.address_space, dtype=torch.float64)
        self.sum = torch.zeros(self.address_space, dtype=torch.float64)
        self.sum_sq = torch.zeros(self.address_space, dtype=torch.float64)
        self.coarse_count = torch.zeros(self.coarse_address_space, dtype=torch.float64)
        self.coarse_sum = torch.zeros(self.coarse_address_space, dtype=torch.float64)
        self.component_address_spaces = self._make_component_address_spaces(population_bins)
        self.component_counts = {
            name: torch.zeros(space, dtype=torch.float64)
            for name, space in self.component_address_spaces.items()
        }
        self.component_sums = {
            name: torch.zeros(space, dtype=torch.float64)
            for name, space in self.component_address_spaces.items()
        }
        self.response = VarianceStats()
        self.candidate = VarianceStats()
        self.background = VarianceStats()
        self._address_mean: Optional[torch.Tensor] = None
        self._coarse_mean: Optional[torch.Tensor] = None
        self._shuffled_address_mean: Optional[torch.Tensor] = None
        self._component_means: Dict[str, torch.Tensor] = {}

    def _make_component_address_spaces(self, population_bins: int) -> Dict[str, int]:
        if self.kind == "token_qk":
            return {
                "token_channel": self.coarse_address_space,
                "plus_q_or_gate": self.coarse_address_space * 2,
                "plus_k": self.address_space,
            }
        pop_factor = int(population_bins) ** 2
        return {
            "token_channel": self.coarse_address_space,
            "plus_q_or_gate": self.address_space // (2 * pop_factor),
            "plus_k": self.address_space // pop_factor,
        }

    def component_address(self, address: torch.Tensor, component: str) -> torch.Tensor:
        if component == "token_channel":
            return address // self.detail_factor
        if self.kind == "token_qk":
            if component == "plus_q_or_gate":
                return address // 2
            if component == "plus_k":
                return address
        else:
            population_factor = self.detail_factor // 4
            if component == "plus_q_or_gate":
                return address // (2 * population_factor)
            if component == "plus_k":
                return address // population_factor
        raise KeyError(f"unknown component address: {component}")

    def update(self, address: torch.Tensor, response: torch.Tensor, candidate_mask: torch.Tensor) -> None:
        addr = address.detach().to("cpu", dtype=torch.long)
        resp = response.detach().to("cpu", dtype=torch.float64)
        cand = candidate_mask.detach().to("cpu", dtype=torch.bool)
        self.count += torch.bincount(addr, minlength=self.address_space).to(torch.float64)
        self.sum += torch.bincount(addr, weights=resp, minlength=self.address_space).to(torch.float64)
        self.sum_sq += torch.bincount(addr, weights=resp * resp, minlength=self.address_space).to(torch.float64)
        coarse_addr = addr // self.detail_factor
        self.coarse_count += torch.bincount(coarse_addr, minlength=self.coarse_address_space).to(torch.float64)
        self.coarse_sum += torch.bincount(
            coarse_addr,
            weights=resp,
            minlength=self.coarse_address_space,
        ).to(torch.float64)
        for name, space in self.component_address_spaces.items():
            component_addr = self.component_address(addr, name)
            self.component_counts[name] += torch.bincount(component_addr, minlength=space).to(torch.float64)
            self.component_sums[name] += torch.bincount(
                component_addr,
                weights=resp,
                minlength=space,
            ).to(torch.float64)
        self.response.update_many(resp.tolist())
        self.candidate.update_many(resp[cand].tolist())
        self.background.update_many(resp[~cand].tolist())
        self._address_mean = None
        self._coarse_mean = None
        self._shuffled_address_mean = None
        self._component_means = {}

    @property
    def global_mean(self) -> float:
        return self.response.mean

    @property
    def candidate_mean(self) -> float:
        return self.candidate.mean if self.candidate.count > 0 else self.global_mean

    @property
    def background_mean(self) -> float:
        return self.background.mean if self.background.count > 0 else self.global_mean

    @property
    def address_mean(self) -> torch.Tensor:
        if self._address_mean is None:
            means = torch.full_like(self.sum, fill_value=float(self.global_mean))
            seen = self.count > 0
            means[seen] = self.sum[seen] / self.count[seen]
            self._address_mean = means
        return self._address_mean

    @property
    def coarse_mean(self) -> torch.Tensor:
        if self._coarse_mean is None:
            means = torch.full_like(self.coarse_sum, fill_value=float(self.global_mean))
            seen = self.coarse_count > 0
            means[seen] = self.coarse_sum[seen] / self.coarse_count[seen]
            self._coarse_mean = means
        return self._coarse_mean

    def component_mean(self, component: str) -> torch.Tensor:
        cached = self._component_means.get(component)
        if cached is not None:
            return cached
        counts = self.component_counts[component]
        sums = self.component_sums[component]
        means = torch.full_like(sums, fill_value=float(self.global_mean))
        seen = counts > 0
        means[seen] = sums[seen] / counts[seen]
        self._component_means[component] = means
        return means

    @property
    def shuffled_address_mean(self) -> torch.Tensor:
        if self._shuffled_address_mean is None:
            means = self.address_mean.clone()
            seen_indices = torch.nonzero(self.count > 0, as_tuple=False).flatten()
            if seen_indices.numel() > 1:
                digest = hashlib.sha256(f"{self.name}:{self.seed}".encode("utf-8")).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                permutation = torch.randperm(seen_indices.numel(), generator=generator)
                means[seen_indices] = means[seen_indices[permutation]]
            self._shuffled_address_mean = means
        return self._shuffled_address_mean

    def summary(self) -> Dict[str, object]:
        seen = self.count > 0
        counts = self.count[seen].tolist()
        unique = int(seen.sum().item())
        singleton = int((self.count == 1).sum().item())
        total = int(self.count.sum().item())
        if counts:
            counts_sorted = sorted(float(x) for x in counts)
            p50 = counts_sorted[len(counts_sorted) // 2]
            p90 = counts_sorted[min(len(counts_sorted) - 1, int(len(counts_sorted) * 0.9))]
            occupancy = {
                "min": float(counts_sorted[0]),
                "p50": float(p50),
                "mean": float(total / unique),
                "p90": float(p90),
                "max": float(counts_sorted[-1]),
            }
        else:
            occupancy = {"min": 0.0, "p50": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
        return {
            "name": self.name,
            "kind": self.kind,
            "stage": self.stage,
            "block": self.block,
            "address_space": self.address_space,
            "token_channel_address_space": self.coarse_address_space,
            "component_address_spaces": dict(self.component_address_spaces),
            "num_samples": total,
            "unique_addresses": unique,
            "address_coverage": unique / float(self.address_space) if self.address_space else None,
            "singleton_fraction": singleton / float(unique) if unique else 0.0,
            "bucket_occupancy": occupancy,
            "response_variance": self.response.variance,
            "candidate_variance": self.candidate.variance,
            "background_variance": self.background.variance,
        }


class ModuleEval:
    def __init__(self, prototype: ModulePrototype) -> None:
        self.prototype = prototype
        self.global_mse = MSEStats()
        self.address_mse = MSEStats()
        self.token_channel_mse = MSEStats()
        self.shuffled_address_mse = MSEStats()
        self.component_mse = {
            "token_channel": MSEStats(),
            "plus_q_or_gate": MSEStats(),
            "plus_k": MSEStats(),
            "full_address": self.address_mse,
            "shuffled_full_address": self.shuffled_address_mse,
        }
        self.candidate_background_mse = MSEStats()
        self.response = VarianceStats()
        self.address_hits = 0
        self.candidates = 0

    def update(self, address: torch.Tensor, response: torch.Tensor, candidate_mask: torch.Tensor) -> None:
        proto = self.prototype
        addr = address.detach().to("cpu", dtype=torch.long)
        resp = response.detach().to("cpu", dtype=torch.float64)
        cand = candidate_mask.detach().to("cpu", dtype=torch.bool)
        count = int(resp.numel())
        if count <= 0:
            return

        global_pred = torch.full_like(resp, fill_value=float(proto.global_mean))
        seen = proto.count[addr] >= proto.min_count
        address_pred = proto.address_mean[addr]
        address_pred = torch.where(seen, address_pred, global_pred)
        coarse_addr = addr // proto.detail_factor
        coarse_seen = proto.coarse_count[coarse_addr] >= proto.min_count
        token_channel_pred = proto.coarse_mean[coarse_addr]
        token_channel_pred = torch.where(coarse_seen, token_channel_pred, global_pred)
        component_predictions = {"token_channel": token_channel_pred}
        for component in ("plus_q_or_gate", "plus_k"):
            component_addr = proto.component_address(addr, component)
            component_seen = proto.component_counts[component][component_addr] >= proto.min_count
            component_pred = proto.component_mean(component)[component_addr]
            component_predictions[component] = torch.where(component_seen, component_pred, global_pred)
        shuffled_address_pred = proto.shuffled_address_mean[addr]
        shuffled_address_pred = torch.where(seen, shuffled_address_pred, global_pred)
        cb_values = torch.where(
            cand,
            torch.full_like(resp, fill_value=float(proto.candidate_mean)),
            torch.full_like(resp, fill_value=float(proto.background_mean)),
        )

        self.global_mse.update(float(torch.sum((resp - global_pred) ** 2).item()), count)
        self.address_mse.update(float(torch.sum((resp - address_pred) ** 2).item()), count)
        self.token_channel_mse.update(float(torch.sum((resp - token_channel_pred) ** 2).item()), count)
        self.shuffled_address_mse.update(float(torch.sum((resp - shuffled_address_pred) ** 2).item()), count)
        for component, pred in component_predictions.items():
            self.component_mse[component].update(float(torch.sum((resp - pred) ** 2).item()), count)
        self.candidate_background_mse.update(float(torch.sum((resp - cb_values) ** 2).item()), count)
        self.response.update_many(resp.tolist())
        self.address_hits += int(seen.sum().item())
        self.candidates += int(cand.sum().item())

    def summary(self) -> Dict[str, object]:
        global_mse = self.global_mse.mse
        address_mse = self.address_mse.mse
        token_channel_mse = self.token_channel_mse.mse
        shuffled_address_mse = self.shuffled_address_mse.mse
        cb_mse = self.candidate_background_mse.mse
        component_values = {
            "component_token_channel_mse": self.component_mse["token_channel"].mse,
            "component_plus_q_or_gate_mse": self.component_mse["plus_q_or_gate"].mse,
            "component_plus_k_mse": self.component_mse["plus_k"].mse,
            "component_full_address_mse": address_mse,
            "component_shuffled_full_address_mse": shuffled_address_mse,
        }
        component_reductions = {
            key.replace("_mse", "_relative_mse_reduction"): (
                (global_mse - value) / global_mse if global_mse > 0 else 0.0
            )
            for key, value in component_values.items()
        }
        return {
            "name": self.prototype.name,
            "kind": self.prototype.kind,
            "stage": self.prototype.stage,
            "block": self.prototype.block,
            "eval_samples": int(self.global_mse.count),
            "eval_response_variance": self.response.variance,
            "eval_address_hit_rate": self.address_hits / float(self.global_mse.count) if self.global_mse.count else 0.0,
            "eval_candidate_fraction": self.candidates / float(self.global_mse.count) if self.global_mse.count else 0.0,
            "global_mean_mse": global_mse,
            "address_lut_mse": address_mse,
            "token_channel_lut_mse": token_channel_mse,
            "shuffled_address_lut_mse": shuffled_address_mse,
            "candidate_background_mse": cb_mse,
            "address_relative_mse_reduction": (
                (global_mse - address_mse) / global_mse if global_mse > 0 else 0.0
            ),
            "token_channel_relative_mse_reduction": (
                (global_mse - token_channel_mse) / global_mse if global_mse > 0 else 0.0
            ),
            "shuffled_address_relative_mse_reduction": (
                (global_mse - shuffled_address_mse) / global_mse if global_mse > 0 else 0.0
            ),
            "candidate_background_relative_mse_reduction": (
                (global_mse - cb_mse) / global_mse if global_mse > 0 else 0.0
            ),
            **component_values,
            **component_reductions,
        }


class PrototypeBank:
    def __init__(self, min_count: int, population_bins: int, seed: int) -> None:
        self.min_count = int(min_count)
        self.population_bins = int(population_bins)
        self.seed = int(seed)
        self.prototypes: Dict[str, ModulePrototype] = {}
        self.eval_stats: Dict[str, ModuleEval] = {}

    def calibrate(self, stats: ModuleDiagnostic, address: torch.Tensor, response: torch.Tensor, candidate_mask: torch.Tensor) -> None:
        proto = self.prototypes.get(stats.name)
        if proto is None:
            proto = ModulePrototype(stats, self.min_count, self.population_bins, self.seed)
            self.prototypes[stats.name] = proto
        proto.update(address, response, candidate_mask)

    def evaluate(self, stats: ModuleDiagnostic, address: torch.Tensor, response: torch.Tensor, candidate_mask: torch.Tensor) -> None:
        proto = self.prototypes.get(stats.name)
        if proto is None:
            return
        evaluator = self.eval_stats.get(stats.name)
        if evaluator is None:
            evaluator = ModuleEval(proto)
            self.eval_stats[stats.name] = evaluator
        evaluator.update(address, response, candidate_mask)

    def calibration_summary(self) -> Dict[str, Dict[str, object]]:
        return {name: proto.summary() for name, proto in self.prototypes.items()}

    def eval_summary(self) -> Dict[str, Dict[str, object]]:
        return {name: evaluator.summary() for name, evaluator in self.eval_stats.items()}


def reset_model_state(model: torch.nn.Module) -> None:
    try:
        from spikingjelly.clock_driven import functional
    except Exception:
        return
    functional.reset_net(model)


def run_pass(
    model: torch.nn.Module,
    loader,
    data_cfg: Dict[str, object],
    diag_cfg: Dict[str, object],
    callback,
    device: torch.device,
) -> Tuple[int, Dict[str, object]]:
    diagnostic = QKAddressDiagnostic(
        model,
        token_bins=int(diag_cfg.get("token_bins", 8)),
        channel_bins=int(diag_cfg.get("channel_bins", 8)),
        population_bins=int(diag_cfg.get("population_bins", 4)),
        max_records_per_module_per_batch=int(diag_cfg.get("max_records_per_module_per_batch", 65536)),
        record_callback=callback,
    )
    max_batches = int(data_cfg["num_batches"])
    processed = 0
    with torch.no_grad():
        for images, _targets in loader:
            if max_batches > 0 and processed >= max_batches:
                break
            if not torch.is_tensor(images):
                images = images[0]
            images = images.to(device, non_blocking=True)
            _ = model(images)
            reset_model_state(model)
            processed += 1
    summary = diagnostic.summary()
    diagnostic.close()
    return processed, summary


def _weighted_mean(items: Iterable[Dict[str, object]], key: str, weight_key: str = "eval_samples") -> Optional[float]:
    total_weight = 0.0
    total = 0.0
    for item in items:
        value = item.get(key)
        weight = float(item.get(weight_key, 0.0))
        if value is None or weight <= 0:
            continue
        total += float(value) * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return total / total_weight


def group_eval_summary(modules: Iterable[Dict[str, object]], key: str) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, list] = {}
    for item in modules:
        grouped.setdefault(str(item[key]), []).append(item)
    component_keys = [
        "component_token_channel_mse",
        "component_plus_q_or_gate_mse",
        "component_plus_k_mse",
        "component_full_address_mse",
        "component_shuffled_full_address_mse",
        "component_token_channel_relative_mse_reduction",
        "component_plus_q_or_gate_relative_mse_reduction",
        "component_plus_k_relative_mse_reduction",
        "component_full_address_relative_mse_reduction",
        "component_shuffled_full_address_relative_mse_reduction",
    ]
    return {
        name: {
            "num_modules": len(items),
            "eval_samples": int(sum(int(item.get("eval_samples", 0)) for item in items)),
            "global_mean_mse": _weighted_mean(items, "global_mean_mse"),
            "address_lut_mse": _weighted_mean(items, "address_lut_mse"),
            "token_channel_lut_mse": _weighted_mean(items, "token_channel_lut_mse"),
            "shuffled_address_lut_mse": _weighted_mean(items, "shuffled_address_lut_mse"),
            "candidate_background_mse": _weighted_mean(items, "candidate_background_mse"),
            "address_relative_mse_reduction": _weighted_mean(items, "address_relative_mse_reduction"),
            "token_channel_relative_mse_reduction": _weighted_mean(
                items, "token_channel_relative_mse_reduction"
            ),
            "shuffled_address_relative_mse_reduction": _weighted_mean(
                items, "shuffled_address_relative_mse_reduction"
            ),
            "candidate_background_relative_mse_reduction": _weighted_mean(items, "candidate_background_relative_mse_reduction"),
            "eval_address_hit_rate": _weighted_mean(items, "eval_address_hit_rate"),
            "eval_candidate_fraction": _weighted_mean(items, "eval_candidate_fraction"),
            **{component_key: _weighted_mean(items, component_key) for component_key in component_keys},
        }
        for name, items in grouped.items()
    }


def write_module_csv(path: Path, modules: Iterable[Dict[str, object]]) -> None:
    rows = list(modules)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


COMPONENT_RECONSTRUCTION_KEYS = [
    "component_token_channel_mse",
    "component_plus_q_or_gate_mse",
    "component_plus_k_mse",
    "component_full_address_mse",
    "component_shuffled_full_address_mse",
    "component_token_channel_relative_mse_reduction",
    "component_plus_q_or_gate_relative_mse_reduction",
    "component_plus_k_relative_mse_reduction",
    "component_full_address_relative_mse_reduction",
    "component_shuffled_full_address_relative_mse_reduction",
]


def run(config_path: Path, output_dir: Path) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(config_path)
    experiment_cfg = dict(cfg["experiment"])
    experiment_seed = int(os.environ.get("QKFORMER_LUT_E1_SEED", experiment_cfg.get("seed", 42)))
    experiment_cfg["seed"] = experiment_seed
    set_seed(experiment_seed)

    model_cfg = dict(cfg["model"])
    diag_cfg = dict(cfg["diagnostic"])
    data_cfg = dict(cfg["data"])
    calibration_cfg = dict(data_cfg["calibration"])
    evaluation_cfg = dict(data_cfg["evaluation"])
    device_name = str(diag_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the upstream cupy-backed QKFormer LIF modules")
    device = torch.device(device_name)

    if model_cfg.get("family") not in {"cifar10", "cifar100"}:
        raise ValueError("E1 currently supports model.family=cifar10 or cifar100")
    env_checkpoint = os.environ.get("QKFORMER_LUT_CKPT")
    if env_checkpoint:
        model_cfg["checkpoint"] = env_checkpoint
    env_time_step = os.environ.get("QKFORMER_LUT_TIME_STEP")
    if env_time_step:
        model_cfg["time_step"] = int(env_time_step)
    env_data_dir = os.environ.get("QKFORMER_LUT_DATA_DIR")
    if env_data_dir:
        calibration_cfg["data_dir"] = env_data_dir
        evaluation_cfg["data_dir"] = env_data_dir
    env_calibration_batches = os.environ.get("QKFORMER_LUT_E1_CALIB_BATCHES")
    if env_calibration_batches is not None:
        calibration_cfg["num_batches"] = int(env_calibration_batches)
    env_evaluation_batches = os.environ.get("QKFORMER_LUT_E1_EVAL_BATCHES")
    if env_evaluation_batches is not None:
        evaluation_cfg["num_batches"] = int(env_evaluation_batches)
    env_calibration_shuffle = os.environ.get("QKFORMER_LUT_E1_CALIB_SHUFFLE")
    if env_calibration_shuffle is not None:
        calibration_cfg["shuffle"] = env_calibration_shuffle.lower() in {"1", "true", "yes", "on"}
    calibration_cfg["seed"] = experiment_seed
    evaluation_cfg["seed"] = experiment_seed

    model = build_cifar10_model(root, model_cfg)
    checkpoint_info = load_checkpoint_if_available(model, model_cfg.get("checkpoint"))
    model.to(device)
    model.eval()

    calibration_loader, calibration_source = build_loader(calibration_cfg, model_cfg, device)
    evaluation_loader, evaluation_source = build_loader(evaluation_cfg, model_cfg, device)

    bank = PrototypeBank(
        min_count=int(diag_cfg.get("prototype_min_count", 2)),
        population_bins=int(diag_cfg.get("population_bins", 4)),
        seed=experiment_seed,
    )
    print("[qk-lut-e1] calibration_start")
    calibration_batches, calibration_hook_summary = run_pass(
        model,
        calibration_loader,
        calibration_cfg,
        diag_cfg,
        bank.calibrate,
        device,
    )
    print(f"[qk-lut-e1] calibration_batches={calibration_batches}")
    print("[qk-lut-e1] evaluation_start")
    evaluation_batches, evaluation_hook_summary = run_pass(
        model,
        evaluation_loader,
        evaluation_cfg,
        diag_cfg,
        bank.evaluate,
        device,
    )
    print(f"[qk-lut-e1] evaluation_batches={evaluation_batches}")

    module_reconstruction = list(bank.eval_summary().values())
    per_stage = group_eval_summary(module_reconstruction, "stage")
    per_block = group_eval_summary(module_reconstruction, "name")
    overall = {
        "eval_samples": int(sum(int(item.get("eval_samples", 0)) for item in module_reconstruction)),
        "global_mean_mse": _weighted_mean(module_reconstruction, "global_mean_mse"),
        "address_lut_mse": _weighted_mean(module_reconstruction, "address_lut_mse"),
        "token_channel_lut_mse": _weighted_mean(module_reconstruction, "token_channel_lut_mse"),
        "shuffled_address_lut_mse": _weighted_mean(module_reconstruction, "shuffled_address_lut_mse"),
        "candidate_background_mse": _weighted_mean(module_reconstruction, "candidate_background_mse"),
        "address_relative_mse_reduction": _weighted_mean(module_reconstruction, "address_relative_mse_reduction"),
        "token_channel_relative_mse_reduction": _weighted_mean(
            module_reconstruction, "token_channel_relative_mse_reduction"
        ),
        "shuffled_address_relative_mse_reduction": _weighted_mean(
            module_reconstruction, "shuffled_address_relative_mse_reduction"
        ),
        "candidate_background_relative_mse_reduction": _weighted_mean(module_reconstruction, "candidate_background_relative_mse_reduction"),
        "eval_address_hit_rate": _weighted_mean(module_reconstruction, "eval_address_hit_rate"),
        "eval_candidate_fraction": _weighted_mean(module_reconstruction, "eval_candidate_fraction"),
        **{
            component_key: _weighted_mean(module_reconstruction, component_key)
            for component_key in COMPONENT_RECONSTRUCTION_KEYS
        },
    }

    verdict = "PENDING_PHASE_GATE_REVIEW"
    if calibration_source == "synthetic" or evaluation_source == "synthetic" or not checkpoint_info["loaded"]:
        verdict = "PENDING_REAL_DATA_CHECKPOINT"

    metrics = {
        "experiment": experiment_cfg,
        "model": {
            "family": model_cfg["family"],
            "time_step": model_cfg["time_step"],
            "layer": model_cfg["layer"],
            "dim": model_cfg["dim"],
            "num_heads": model_cfg["num_heads"],
            "checkpoint": checkpoint_info,
        },
        "data": {
            "calibration": {
                "source": calibration_source,
                "split": calibration_cfg.get("split"),
                "num_batches": calibration_batches,
                "batch_size": calibration_cfg["batch_size"],
                "shuffle": bool(calibration_cfg.get("shuffle", False)),
                "seed": int(calibration_cfg.get("seed", experiment_seed)),
            },
            "evaluation": {
                "source": evaluation_source,
                "split": evaluation_cfg.get("split"),
                "num_batches": evaluation_batches,
                "batch_size": evaluation_cfg["batch_size"],
                "shuffle": bool(evaluation_cfg.get("shuffle", False)),
                "seed": int(evaluation_cfg.get("seed", experiment_seed)),
            },
        },
        "diagnostic_config": diag_cfg,
        "verdict": verdict,
        "overall_reconstruction": overall,
        "module_reconstruction": module_reconstruction,
        "per_stage_reconstruction": per_stage,
        "per_block_reconstruction": per_block,
        "calibration_prototypes": bank.calibration_summary(),
        "calibration_hook_summary": calibration_hook_summary,
        "evaluation_hook_summary": evaluation_hook_summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    write_module_csv(output_dir / "module_reconstruction.csv", module_reconstruction)
    print(f"[qk-lut-e1] wrote {metrics_path}")
    print(f"[qk-lut-e1] verdict={verdict}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QK-LUTFormer E1 split-aware reconstruction diagnostic")
    parser.add_argument("--config", type=Path, default=Path("configs/qkformer_lut_e1_recon.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
