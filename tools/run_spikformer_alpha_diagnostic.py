#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from qkformer_lut.cross_arch_hooks import (
    DensePrototypeLUT,
    _restore_spikformer,
    _spikformer_to_tbnc,
    freeze_backbone,
    spikformer_qk_scalar_address,
)


MODE_LABELS = {
    "aligned_lut": "Aligned LUT",
    "shuffled_address": "Shuffled Address Control",
    "token_channel_mean": "Token/Channel Mean Control",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_snn(model: nn.Module) -> None:
    try:
        from spikingjelly.clock_driven import functional

        functional.reset_net(model)
    except Exception:
        pass


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get("state_dict") or payload.get("model") or payload
    else:
        state = payload
    return {str(k).replace("module.", "", 1): v for k, v in state.items() if torch.is_tensor(v)}


def build_spikformer(repo: Path, num_classes: int, checkpoint: Path) -> Tuple[nn.Module, Dict[str, object]]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(repo / "cifar10"))
    try:
        model_module = importlib.import_module("model")
        model = model_module.Spikformer(
            img_size_h=32,
            img_size_w=32,
            patch_size=4,
            in_channels=3,
            num_classes=num_classes,
            embed_dims=384,
            num_heads=12,
            mlp_ratios=4,
            qkv_bias=False,
            depths=4,
            sr_ratios=1,
            T=4,
        )
    finally:
        try:
            sys.path.remove(str(repo / "cifar10"))
        except ValueError:
            pass
    msg = model.load_state_dict(load_state_dict(checkpoint), strict=False)
    return model, {
        "checkpoint": str(checkpoint),
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }


def cifar_loader(data_root: Path, split: str, batch_size: int, workers: int):
    ds = datasets.CIFAR10(
        root=str(data_root),
        train=split == "train",
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        ),
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@dataclass
class ScalarMeter:
    total: float = 0.0
    weight: int = 0

    def update(self, value: float, weight: int = 1) -> None:
        if math.isfinite(float(value)) and weight > 0:
            self.total += float(value) * int(weight)
            self.weight += int(weight)

    def mean(self) -> float:
        return self.total / self.weight if self.weight else 0.0


@dataclass
class TensorMoment:
    sum: Optional[torch.Tensor] = None
    sumsq: Optional[torch.Tensor] = None
    count: int = 0

    @torch.no_grad()
    def update_tbnc(self, x: torch.Tensor) -> None:
        value = x.detach().to(torch.float64)
        reduce_dims = (1, 2)
        current_sum = value.sum(dim=reduce_dims).cpu()
        current_sumsq = (value * value).sum(dim=reduce_dims).cpu()
        current_count = int(value.shape[1] * value.shape[2])
        if self.sum is None:
            self.sum = current_sum
            self.sumsq = current_sumsq
        else:
            self.sum += current_sum
            self.sumsq += current_sumsq
        self.count += current_count

    def mean_std(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sum is None or self.sumsq is None or self.count <= 0:
            raise RuntimeError("moment statistics are empty")
        mean = self.sum / float(self.count)
        var = torch.clamp(self.sumsq / float(self.count) - mean * mean, min=1.0e-12)
        return mean.to(device=device, dtype=dtype), var.sqrt().to(device=device, dtype=dtype)


@dataclass
class LocalMeters:
    local_mse: ScalarMeter = field(default_factory=ScalarMeter)
    local_cosine: ScalarMeter = field(default_factory=ScalarMeter)
    orig_rms: ScalarMeter = field(default_factory=ScalarMeter)
    lut_rms: ScalarMeter = field(default_factory=ScalarMeter)
    rms_ratio: ScalarMeter = field(default_factory=ScalarMeter)
    orig_std: ScalarMeter = field(default_factory=ScalarMeter)
    lut_std: ScalarMeter = field(default_factory=ScalarMeter)
    std_ratio: ScalarMeter = field(default_factory=ScalarMeter)
    output_zero_frac: ScalarMeter = field(default_factory=ScalarMeter)
    hit_rate: ScalarMeter = field(default_factory=ScalarMeter)
    lookup_count: int = 0
    hit_count: int = 0
    spike_zero_count: int = 0
    spike_value_count: int = 0
    per_t_orig_mean: Dict[int, ScalarMeter] = field(default_factory=lambda: defaultdict(ScalarMeter))
    per_t_lut_mean: Dict[int, ScalarMeter] = field(default_factory=lambda: defaultdict(ScalarMeter))
    per_t_blend_mean: Dict[int, ScalarMeter] = field(default_factory=lambda: defaultdict(ScalarMeter))
    per_t_orig_sparsity: Dict[int, ScalarMeter] = field(default_factory=lambda: defaultdict(ScalarMeter))
    per_t_lut_sparsity: Dict[int, ScalarMeter] = field(default_factory=lambda: defaultdict(ScalarMeter))
    per_t_blend_sparsity: Dict[int, ScalarMeter] = field(default_factory=lambda: defaultdict(ScalarMeter))

    def summary(self) -> Dict[str, object]:
        return {
            "local_mse": self.local_mse.mean(),
            "local_cosine": self.local_cosine.mean(),
            "orig_rms": self.orig_rms.mean(),
            "lut_rms": self.lut_rms.mean(),
            "orig_lut_rms_ratio": self.rms_ratio.mean(),
            "orig_std": self.orig_std.mean(),
            "lut_std": self.lut_std.mean(),
            "orig_lut_std_ratio": self.std_ratio.mean(),
            "output_zero_fraction": self.output_zero_frac.mean(),
            "spike_sparsity": self.spike_zero_count / self.spike_value_count if self.spike_value_count else 0.0,
            "lut_coverage_hit_rate": self.hit_count / self.lookup_count if self.lookup_count else 0.0,
            "fallback_rate": 1.0 - (self.hit_count / self.lookup_count) if self.lookup_count else 1.0,
            "per_time": {
                str(t): {
                    "orig_mean": self.per_t_orig_mean[t].mean(),
                    "lut_mean": self.per_t_lut_mean[t].mean(),
                    "blend_mean": self.per_t_blend_mean[t].mean(),
                    "orig_sparsity": self.per_t_orig_sparsity[t].mean(),
                    "lut_sparsity": self.per_t_lut_sparsity[t].mean(),
                    "blend_sparsity": self.per_t_blend_sparsity[t].mean(),
                }
                for t in sorted(set(self.per_t_orig_mean) | set(self.per_t_lut_mean) | set(self.per_t_blend_mean))
            },
        }


class SpikformerAlphaLUTHook:
    def __init__(
        self,
        model: nn.Module,
        mode: str,
        token_bins: int = 8,
        channel_bins: int = 8,
        population_bins: int = 4,
        min_support: int = 2,
        seed: int = 42,
    ) -> None:
        self.model = freeze_backbone(model)
        self.mode = mode
        self.token_bins = int(token_bins)
        self.channel_bins = int(channel_bins)
        self.population_bins = int(population_bins)
        self.min_support = int(min_support)
        self.seed = int(seed)
        self.enabled = False
        self.collect_calibration = False
        self.collect_lut_moments = False
        self.alpha = 0.0
        self.moment_match = False
        self.buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.luts: Dict[str, DensePrototypeLUT] = {}
        self.orig_moments: Dict[str, TensorMoment] = defaultdict(TensorMoment)
        self.lut_moments: Dict[str, TensorMoment] = defaultdict(TensorMoment)
        self.device_lut_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
        self.metrics = LocalMeters()
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset_metrics(self) -> None:
        self.metrics = LocalMeters()

    def finalize(self) -> None:
        for lut in self.luts.values():
            lut.finalize()
        self.device_lut_cache.clear()

    def supported_scalar_entries(self) -> int:
        return sum(lut.supported_entries for lut in self.luts.values())

    def memory_kib(self) -> float:
        return self.supported_scalar_entries() * 4.0 / 1024.0

    def set_eval_condition(self, alpha: float, moment_match: bool = False) -> None:
        self.collect_calibration = False
        self.collect_lut_moments = False
        self.enabled = True
        self.alpha = float(alpha)
        self.moment_match = bool(moment_match)
        self.reset_metrics()

    def set_passthrough(self) -> None:
        self.collect_calibration = False
        self.collect_lut_moments = False
        self.enabled = False
        self.alpha = 0.0
        self.moment_match = False

    def _register(self) -> None:
        for name, module in self.model.named_modules():
            if module.__class__.__name__ != "SSA":
                continue
            for child_name in ("q_lif", "k_lif", "proj_lif"):
                child = getattr(module, child_name, None)
                if child is not None:
                    self.handles.append(child.register_forward_hook(self._make_hook(name, child_name[:-4])))

    def _make_hook(self, prefix: str, kind: str):
        def hook(_module, _inputs, output):
            self.buffers[prefix][kind] = output.detach()
            if kind in {"q", "k"} and (self.enabled or self.collect_calibration):
                binary = output.detach() <= 0
                self.metrics.spike_zero_count += int(binary.sum().item())
                self.metrics.spike_value_count += int(binary.numel())
            if kind == "proj":
                return self._consume(prefix, output)
            return None

        return hook

    @torch.no_grad()
    def _address(self, prefix: str, proj_output: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor, str, Tuple[int, ...]]:
        module = self.model.get_submodule(prefix)
        buf = self.buffers[prefix]
        if "q" not in buf or "k" not in buf:
            raise RuntimeError(f"missing q/k buffer for {prefix}")
        dim = int(getattr(module, "dim"))
        q_tbnc, _, _ = _spikformer_to_tbnc(buf["q"], dim)
        k_tbnc, _, _ = _spikformer_to_tbnc(buf["k"], dim)
        proj_tbnc, layout, shape = _spikformer_to_tbnc(proj_output, dim)
        address, coarse_address, address_space, coarse_address_space = spikformer_qk_scalar_address(
            q_tbnc=q_tbnc,
            k_tbnc=k_tbnc,
            num_heads=int(getattr(module, "num_heads")),
            token_bins=self.token_bins,
            channel_bins=self.channel_bins,
            population_bins=self.population_bins,
        )
        if self.mode == "token_channel_mean":
            address = coarse_address
            address_space = coarse_address_space
        return address, address_space, proj_tbnc, layout, shape

    @torch.no_grad()
    def _consume(self, prefix: str, proj_output: torch.Tensor) -> Optional[torch.Tensor]:
        if "q" not in self.buffers[prefix] or "k" not in self.buffers[prefix]:
            return None
        address, address_space, proj_tbnc, layout, shape = self._address(prefix, proj_output)
        lut = self.luts.get(prefix)
        if lut is None:
            lut = DensePrototypeLUT(address_space, mode=self.mode, min_support=self.min_support, seed=self.seed)
            self.luts[prefix] = lut
        if self.collect_calibration:
            lut.update(address, proj_tbnc)
            self.orig_moments[prefix].update_tbnc(proj_tbnc)
            return None
        if self.collect_lut_moments:
            pred, _seen = self._predict(prefix, address)
            self.lut_moments[prefix].update_tbnc(pred.reshape_as(proj_tbnc))
            return None
        if not self.enabled:
            return None

        pred, seen = self._predict(prefix, address)
        lut_tbnc = pred.reshape_as(proj_tbnc).to(dtype=proj_tbnc.dtype)
        if self.moment_match:
            lut_tbnc = self._moment_match(prefix, lut_tbnc)
        blended = proj_tbnc + self.alpha * (lut_tbnc - proj_tbnc)
        self._update_local_metrics(proj_tbnc, lut_tbnc, blended, seen)
        return _restore_spikformer(blended, layout, shape).to(proj_output.dtype)

    @torch.no_grad()
    def _predict(self, prefix: str, address: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lut = self.luts[prefix]
        if lut.mean is None or lut.support is None:
            raise RuntimeError("call finalize() before predict()")
        key = (prefix, str(address.device))
        cache = self.device_lut_cache.get(key)
        if cache is None:
            cache = {
                "mean": lut.mean.to(address.device),
                "support": lut.support.to(address.device),
                "global_mean": torch.tensor(float(lut.global_mean.item()), device=address.device, dtype=torch.float32),
            }
            if self.mode == "shuffled_address":
                gen = torch.Generator(device="cpu").manual_seed(self.seed)
                cache["perm"] = torch.randperm(lut.address_space, generator=gen, dtype=torch.long).to(address.device)
            self.device_lut_cache[key] = cache
        query = address.long()
        if self.mode == "shuffled_address":
            query = cache["perm"][query]
        pred = cache["mean"][query]
        seen = cache["support"][query]
        fallback = torch.full_like(pred, float(cache["global_mean"].item()))
        pred = torch.where(seen, pred, fallback)
        return pred, seen

    @torch.no_grad()
    def _moment_match(self, prefix: str, lut_tbnc: torch.Tensor) -> torch.Tensor:
        if prefix not in self.orig_moments or prefix not in self.lut_moments:
            return lut_tbnc
        orig_mean, orig_std = self.orig_moments[prefix].mean_std(lut_tbnc.device, lut_tbnc.dtype)
        lut_mean, lut_std = self.lut_moments[prefix].mean_std(lut_tbnc.device, lut_tbnc.dtype)
        view_shape = (orig_mean.shape[0], 1, 1, orig_mean.shape[1])
        return (lut_tbnc - lut_mean.reshape(view_shape)) / (lut_std.reshape(view_shape) + 1.0e-6) * orig_std.reshape(view_shape) + orig_mean.reshape(view_shape)

    @torch.no_grad()
    def _update_local_metrics(self, orig: torch.Tensor, lut: torch.Tensor, blended: torch.Tensor, seen: torch.Tensor) -> None:
        n = int(orig.numel())
        orig_f = orig.float()
        lut_f = lut.float()
        blend_f = blended.float()
        mse = F.mse_loss(lut_f, orig_f, reduction="mean").item()
        cosine = F.cosine_similarity(orig_f.reshape(1, -1), lut_f.reshape(1, -1), dim=1).item()
        orig_rms = float(torch.sqrt(torch.mean(orig_f * orig_f) + 1.0e-12).item())
        lut_rms = float(torch.sqrt(torch.mean(lut_f * lut_f) + 1.0e-12).item())
        orig_std = float(orig_f.std(unbiased=False).item())
        lut_std = float(lut_f.std(unbiased=False).item())
        self.metrics.local_mse.update(mse, n)
        self.metrics.local_cosine.update(cosine, n)
        self.metrics.orig_rms.update(orig_rms, n)
        self.metrics.lut_rms.update(lut_rms, n)
        self.metrics.rms_ratio.update(lut_rms / max(orig_rms, 1.0e-12), n)
        self.metrics.orig_std.update(orig_std, n)
        self.metrics.lut_std.update(lut_std, n)
        self.metrics.std_ratio.update(lut_std / max(orig_std, 1.0e-12), n)
        self.metrics.output_zero_frac.update(float((blend_f.abs() < 1.0e-8).float().mean().item()), n)
        hit = seen.detach().bool()
        self.metrics.hit_count += int(hit.sum().item())
        self.metrics.lookup_count += int(hit.numel())
        self.metrics.hit_rate.update(float(hit.float().mean().item()), int(hit.numel()))
        for t in range(orig.shape[0]):
            weight = int(orig[t].numel())
            self.metrics.per_t_orig_mean[t].update(float(orig_f[t].mean().item()), weight)
            self.metrics.per_t_lut_mean[t].update(float(lut_f[t].mean().item()), weight)
            self.metrics.per_t_blend_mean[t].update(float(blend_f[t].mean().item()), weight)
            self.metrics.per_t_orig_sparsity[t].update(float((orig_f[t].abs() < 1.0e-8).float().mean().item()), weight)
            self.metrics.per_t_lut_sparsity[t].update(float((lut_f[t].abs() < 1.0e-8).float().mean().item()), weight)
            self.metrics.per_t_blend_sparsity[t].update(float((blend_f[t].abs() < 1.0e-8).float().mean().item()), weight)


@dataclass
class ProbeMeters:
    bn_abs_z: ScalarMeter = field(default_factory=ScalarMeter)
    bn_var_ratio: ScalarMeter = field(default_factory=ScalarMeter)
    lif_spike_rate: ScalarMeter = field(default_factory=ScalarMeter)
    lif_near_threshold: ScalarMeter = field(default_factory=ScalarMeter)
    lif_margin_abs: ScalarMeter = field(default_factory=ScalarMeter)

    def summary(self) -> Dict[str, float]:
        return {
            "bn_mean_abs_z_shift": self.bn_abs_z.mean(),
            "bn_var_ratio": self.bn_var_ratio.mean(),
            "lif_spike_rate": self.lif_spike_rate.mean(),
            "lif_near_threshold_frac": self.lif_near_threshold.mean(),
            "lif_threshold_margin_abs": self.lif_margin_abs.mean(),
        }


class DownstreamProbe:
    def __init__(self, model: nn.Module, threshold_eps: float = 0.05) -> None:
        self.model = model
        self.threshold_eps = float(threshold_eps)
        self.context = "clean"
        self.meters: Dict[str, ProbeMeters] = defaultdict(ProbeMeters)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self) -> None:
        self.meters = defaultdict(ProbeMeters)

    def set_context(self, context: str) -> None:
        self.context = context

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {key: meter.summary() for key, meter in self.meters.items()}

    def _register(self) -> None:
        for _name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.handles.append(module.register_forward_pre_hook(self._bn_pre_hook(module)))
            if "LIFNode" in module.__class__.__name__ or "IFNode" in module.__class__.__name__:
                self.handles.append(module.register_forward_hook(self._lif_hook(module)))

    def _bn_pre_hook(self, module: nn.Module):
        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            x = inputs[0].detach().float()
            running_mean = getattr(module, "running_mean", None)
            running_var = getattr(module, "running_var", None)
            if running_mean is None or running_var is None:
                return None
            c = int(running_mean.numel())
            channel_dim = None
            for dim, size in enumerate(x.shape):
                if int(size) == c:
                    channel_dim = dim
                    break
            if channel_dim is None:
                return None
            reduce_dims = tuple(i for i in range(x.ndim) if i != channel_dim)
            batch_mean = x.mean(dim=reduce_dims)
            batch_var = x.var(dim=reduce_dims, unbiased=False)
            rm = running_mean.detach().to(device=x.device, dtype=x.dtype)
            rv = running_var.detach().to(device=x.device, dtype=x.dtype).clamp_min(1.0e-8)
            z = torch.abs(batch_mean - rm) / torch.sqrt(rv)
            ratio = batch_var / rv
            weight = int(batch_mean.numel())
            self.meters[self.context].bn_abs_z.update(float(z.mean().item()), weight)
            self.meters[self.context].bn_var_ratio.update(float(ratio.mean().item()), weight)
            return None

        return hook

    def _lif_hook(self, module: nn.Module):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return None
            y = output.detach().float()
            self.meters[self.context].lif_spike_rate.update(float((y > 0).float().mean().item()), int(y.numel()))
            v = getattr(module, "v", None)
            if torch.is_tensor(v):
                v_f = v.detach().float()
                threshold = float(getattr(module, "v_threshold", 1.0))
                margin = torch.abs(threshold - v_f)
                self.meters[self.context].lif_near_threshold.update(float((margin < self.threshold_eps).float().mean().item()), int(v_f.numel()))
                self.meters[self.context].lif_margin_abs.update(float(margin.mean().item()), int(v_f.numel()))
            return None

        return hook


@torch.no_grad()
def top1(logits: torch.Tensor, target: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == target).sum().item())


@torch.no_grad()
def calibrate(model: nn.Module, loader, device: torch.device, hook: SpikformerAlphaLUTHook, max_batches: int) -> int:
    hook.collect_calibration = True
    hook.collect_lut_moments = False
    hook.enabled = False
    model.eval()
    batches = 0
    for images, _target in loader:
        if batches >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        reset_snn(model)
        _ = model(images)
        batches += 1
    reset_snn(model)
    hook.collect_calibration = False
    hook.finalize()
    return batches


@torch.no_grad()
def collect_lut_moments(model: nn.Module, loader, device: torch.device, hook: SpikformerAlphaLUTHook, max_batches: int) -> int:
    hook.collect_lut_moments = True
    hook.collect_calibration = False
    hook.enabled = False
    model.eval()
    batches = 0
    for images, _target in loader:
        if batches >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        reset_snn(model)
        _ = model(images)
        batches += 1
    reset_snn(model)
    hook.collect_lut_moments = False
    return batches


@torch.no_grad()
def evaluate_condition(
    model: nn.Module,
    loader,
    device: torch.device,
    hook: SpikformerAlphaLUTHook,
    probe: DownstreamProbe,
    alpha: float,
    max_batches: Optional[int],
    moment_match: bool = False,
) -> Dict[str, object]:
    model.eval()
    hook.set_eval_condition(alpha=alpha, moment_match=moment_match)
    probe.reset()
    clean_correct = 0
    injected_correct = 0
    total = 0
    kl_meter = ScalarMeter()
    logit_mse_meter = ScalarMeter()
    logit_cos_meter = ScalarMeter()
    for idx, (images, target) in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        hook.set_passthrough()
        probe.set_context("clean")
        reset_snn(model)
        clean_logits = model(images)

        hook.set_eval_condition(alpha=alpha, moment_match=moment_match)
        probe.set_context("injected")
        reset_snn(model)
        injected_logits = model(images)

        clean_correct += top1(clean_logits, target)
        injected_correct += top1(injected_logits, target)
        total += int(target.numel())
        clean_logp = F.log_softmax(clean_logits.float(), dim=1)
        injected_logp = F.log_softmax(injected_logits.float(), dim=1)
        clean_p = clean_logp.exp()
        kl = F.kl_div(injected_logp, clean_p, reduction="batchmean").item()
        mse = F.mse_loss(injected_logits.float(), clean_logits.float(), reduction="mean").item()
        cos = F.cosine_similarity(clean_logits.float(), injected_logits.float(), dim=1).mean().item()
        kl_meter.update(kl, int(target.numel()))
        logit_mse_meter.update(mse, int(target.numel()))
        logit_cos_meter.update(cos, int(target.numel()))

    reset_snn(model)
    local = hook.metrics.summary()
    probe_summary = probe.summary()
    return {
        "alpha": float(alpha),
        "moment_match": bool(moment_match),
        "clean_top1": 100.0 * clean_correct / max(total, 1),
        "injected_top1": 100.0 * injected_correct / max(total, 1),
        "drop_top1": 100.0 * (clean_correct - injected_correct) / max(total, 1),
        "num_eval_samples": total,
        "kl_clean_to_injected": kl_meter.mean(),
        "logit_mse": logit_mse_meter.mean(),
        "logit_cosine": logit_cos_meter.mean(),
        **local,
        "probe": probe_summary,
    }


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader, device: torch.device, hook: SpikformerAlphaLUTHook, max_batches: int) -> int:
    for param in model.parameters():
        param.requires_grad_(False)
    hook.set_eval_condition(alpha=1.0, moment_match=False)
    model.train()
    batches = 0
    for images, _target in loader:
        if batches >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        reset_snn(model)
        _ = model(images)
        batches += 1
    reset_snn(model)
    model.eval()
    return batches


def parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def run(args) -> Dict[str, object]:
    set_seed(args.seed)
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.local_cuda_index}" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    calib_loader = cifar_loader(data_root, "train", args.batch_size, args.workers)
    val_loader = cifar_loader(data_root, "validation", args.batch_size, args.workers)
    rows: List[Dict[str, object]] = []
    detailed: List[Dict[str, object]] = []
    partial_csv = result_dir / "summary_partial.csv"
    partial_jsonl = result_dir / "rows_partial.jsonl"

    model, load_report = build_spikformer(repo_root, 10, checkpoint)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    mode_to_alphas = {
        "aligned_lut": parse_float_list(args.aligned_alphas),
        "shuffled_address": parse_float_list(args.control_alphas),
        "token_channel_mean": parse_float_list(args.control_alphas),
    }
    calibration_reports: Dict[str, object] = {}

    for mode, alphas in mode_to_alphas.items():
        hook = SpikformerAlphaLUTHook(model, mode=mode, min_support=args.min_support, seed=args.seed)
        probe = DownstreamProbe(model, threshold_eps=args.threshold_eps)
        calib_batches = calibrate(model, calib_loader, device, hook, args.calib_batches)
        moment_batches = collect_lut_moments(model, calib_loader, device, hook, args.calib_batches)
        calibration_reports[mode] = {
            "label": MODE_LABELS[mode],
            "calib_batches": calib_batches,
            "moment_batches": moment_batches,
            "supported_scalar_entries": hook.supported_scalar_entries(),
            "analytical_memory_footprint_kib": hook.memory_kib(),
            "num_lut_modules": len(hook.luts),
        }
        for alpha in alphas:
            row = evaluate_condition(model, val_loader, device, hook, probe, alpha, args.max_eval_batches, moment_match=False)
            row.update(
                {
                    "mode": mode,
                    "mode_label": MODE_LABELS[mode],
                    "variant": "standard",
                    "supported_scalar_entries": hook.supported_scalar_entries(),
                    "analytical_memory_footprint_kib": hook.memory_kib(),
                }
            )
            detailed.append(row)
            flat = flatten_row(row)
            rows.append(flat)
            append_partial(partial_csv, partial_jsonl, flat, row)
            print(
                f"[condition] mode={mode} variant=standard alpha={alpha} "
                f"top1={row['injected_top1']:.4f} drop={row['drop_top1']:.4f}",
                flush=True,
            )
        if mode == "aligned_lut" and args.run_moment_match:
            row = evaluate_condition(model, val_loader, device, hook, probe, 1.0, args.max_eval_batches, moment_match=True)
            row.update(
                {
                    "mode": mode,
                    "mode_label": MODE_LABELS[mode],
                    "variant": "moment_matched_hard",
                    "moment_match_scope": "calibration_per_time_per_channel",
                    "supported_scalar_entries": hook.supported_scalar_entries(),
                    "analytical_memory_footprint_kib": hook.memory_kib(),
                }
            )
            detailed.append(row)
            flat = flatten_row(row)
            rows.append(flat)
            append_partial(partial_csv, partial_jsonl, flat, row)
            print(
                f"[condition] mode={mode} variant=moment_matched_hard alpha=1.0 "
                f"top1={row['injected_top1']:.4f} drop={row['drop_top1']:.4f}",
                flush=True,
            )
        hook.close()
        probe.close()

    if args.run_bn_recalibration:
        bn_model, _bn_load_report = build_spikformer(repo_root, 10, checkpoint)
        bn_model.to(device)
        for p in bn_model.parameters():
            p.requires_grad_(False)
        bn_hook = SpikformerAlphaLUTHook(bn_model, mode="aligned_lut", min_support=args.min_support, seed=args.seed)
        bn_probe = DownstreamProbe(bn_model, threshold_eps=args.threshold_eps)
        bn_calib_batches = calibrate(bn_model, calib_loader, device, bn_hook, args.calib_batches)
        bn_moment_batches = collect_lut_moments(bn_model, calib_loader, device, bn_hook, args.calib_batches)
        bn_recal_batches = recalibrate_bn(bn_model, calib_loader, device, bn_hook, args.bn_recal_batches)
        row = evaluate_condition(bn_model, val_loader, device, bn_hook, bn_probe, 1.0, args.max_eval_batches, moment_match=False)
        row.update(
            {
                "mode": "aligned_lut",
                "mode_label": MODE_LABELS["aligned_lut"],
                "variant": "bn_recalibrated_hard",
                "bn_recalibration_batches": bn_recal_batches,
                "calib_batches": bn_calib_batches,
                "moment_batches": bn_moment_batches,
                "supported_scalar_entries": bn_hook.supported_scalar_entries(),
                "analytical_memory_footprint_kib": bn_hook.memory_kib(),
            }
        )
        detailed.append(row)
        flat = flatten_row(row)
        rows.append(flat)
        append_partial(partial_csv, partial_jsonl, flat, row)
        print(
            f"[condition] mode=aligned_lut variant=bn_recalibrated_hard alpha=1.0 "
            f"top1={row['injected_top1']:.4f} drop={row['drop_top1']:.4f}",
            flush=True,
        )
        bn_hook.close()
        bn_probe.close()

    payload = {
        "experiment": "spikformer_alpha_blending_injection_diagnostic",
        "track": "spikformer",
        "dataset": "cifar10",
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "checkpoint": str(checkpoint),
        "load_report": load_report,
        "args": vars(args),
        "calibration_reports": calibration_reports,
        "rows": detailed,
    }
    write_json(result_dir / "metrics.json", payload)
    write_csv(result_dir / "summary.csv", rows)
    return payload


def flatten_row(row: Dict[str, object]) -> Dict[str, object]:
    probe = row.get("probe", {})
    clean_probe = probe.get("clean", {}) if isinstance(probe, dict) else {}
    injected_probe = probe.get("injected", {}) if isinstance(probe, dict) else {}
    return {
        "mode": row.get("mode"),
        "variant": row.get("variant"),
        "alpha": row.get("alpha"),
        "moment_match": row.get("moment_match"),
        "clean_top1": row.get("clean_top1"),
        "injected_top1": row.get("injected_top1"),
        "drop_top1": row.get("drop_top1"),
        "kl_clean_to_injected": row.get("kl_clean_to_injected"),
        "logit_mse": row.get("logit_mse"),
        "logit_cosine": row.get("logit_cosine"),
        "local_mse": row.get("local_mse"),
        "local_cosine": row.get("local_cosine"),
        "orig_lut_rms_ratio": row.get("orig_lut_rms_ratio"),
        "orig_lut_std_ratio": row.get("orig_lut_std_ratio"),
        "spike_sparsity": row.get("spike_sparsity"),
        "lut_coverage_hit_rate": row.get("lut_coverage_hit_rate"),
        "fallback_rate": row.get("fallback_rate"),
        "analytical_memory_footprint_kib": row.get("analytical_memory_footprint_kib"),
        "supported_scalar_entries": row.get("supported_scalar_entries"),
        "clean_bn_z": clean_probe.get("bn_mean_abs_z_shift"),
        "injected_bn_z": injected_probe.get("bn_mean_abs_z_shift"),
        "clean_bn_var_ratio": clean_probe.get("bn_var_ratio"),
        "injected_bn_var_ratio": injected_probe.get("bn_var_ratio"),
        "clean_lif_spike_rate": clean_probe.get("lif_spike_rate"),
        "injected_lif_spike_rate": injected_probe.get("lif_spike_rate"),
        "clean_lif_near_threshold": clean_probe.get("lif_near_threshold_frac"),
        "injected_lif_near_threshold": injected_probe.get("lif_near_threshold_frac"),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_partial(csv_path: Path, jsonl_path: Path, flat_row: Dict[str, object], full_row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat_row)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(full_row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-cuda-index", type=int, default=0)
    parser.add_argument("--aligned-alphas", default="0.0,0.05,0.10,0.25,0.50,1.00")
    parser.add_argument("--control-alphas", default="0.10,0.25,1.00")
    parser.add_argument("--threshold-eps", type=float, default=0.05)
    parser.add_argument("--run-moment-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-bn-recalibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bn-recal-batches", type=int, default=196)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"result_dir": str(Path(args.result_dir).resolve()), "rows": len(payload["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
