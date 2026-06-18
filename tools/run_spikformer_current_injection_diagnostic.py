#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from qkformer_lut.cross_arch_hooks import DensePrototypeLUT, spikformer_qk_scalar_address
from tools.run_spikformer_alpha_diagnostic import (
    DownstreamProbe,
    LocalMeters,
    ScalarMeter,
    TensorMoment,
    build_spikformer,
    cifar_loader,
    reset_snn,
    set_seed,
    write_json,
)


def to_tbnc_from_proj_linear(x: torch.Tensor, q_like: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if q_like.ndim != 4:
        raise ValueError(f"expected q_lif tensor [T,B,N,C], got {tuple(q_like.shape)}")
    t, b, n, c = q_like.shape
    if tuple(x.shape) != (t * b, n, c):
        raise ValueError(f"expected proj_linear output {(t * b, n, c)}, got {tuple(x.shape)}")
    return x.reshape(t, b, n, c).contiguous(), tuple(x.shape)


def restore_proj_linear(x_tbnc: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    return x_tbnc.reshape(shape)


class GateBank(nn.Module):
    def __init__(self, modules: List[str], dim: int, variant: str, init_temporal: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if variant not in {"none", "static_t", "learned_t", "learned_tc"}:
            raise ValueError(variant)
        self.variant = variant
        self.dim = int(dim)
        self.module_to_key = {name: name.replace(".", "__") for name in modules}
        for name in modules:
            key = self.module_to_key[name]
            if variant in {"learned_t", "learned_tc"}:
                init = init_temporal.detach().float().clone() if init_temporal is not None else torch.ones(4)
                self.register_parameter(f"{key}_gamma_t", nn.Parameter(init))
            elif variant == "static_t":
                self.register_buffer(f"{key}_gamma_t", init_temporal.detach().float().clone() if init_temporal is not None else torch.ones(4))
            if variant == "learned_tc":
                self.register_parameter(f"{key}_gamma_c", nn.Parameter(torch.ones(dim)))
                self.register_parameter(f"{key}_beta_c", nn.Parameter(torch.zeros(dim)))

    def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "none":
            return x
        key = self.module_to_key[name]
        gamma_t = getattr(self, f"{key}_gamma_t").to(device=x.device, dtype=x.dtype).reshape(-1, 1, 1, 1)
        out = gamma_t * x
        if self.variant == "learned_tc":
            gamma_c = getattr(self, f"{key}_gamma_c").to(device=x.device, dtype=x.dtype).reshape(1, 1, 1, -1)
            beta_c = getattr(self, f"{key}_beta_c").to(device=x.device, dtype=x.dtype).reshape(1, 1, 1, -1)
            out = gamma_c * out + beta_c
        return out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CurrentLUTHook:
    def __init__(
        self,
        model: nn.Module,
        mode: str = "aligned_lut",
        token_bins: int = 8,
        channel_bins: int = 8,
        population_bins: int = 4,
        min_support: int = 2,
        seed: int = 42,
        gate_bank: Optional[GateBank] = None,
    ) -> None:
        self.model = model
        self.mode = mode
        self.token_bins = int(token_bins)
        self.channel_bins = int(channel_bins)
        self.population_bins = int(population_bins)
        self.min_support = int(min_support)
        self.seed = int(seed)
        self.gate_bank = gate_bank
        self.enabled = False
        self.collect_calibration = False
        self.collect_lut_moments = False
        self.collect_clean_spikes = False
        self.alpha = 1.0
        self.moment_match = False
        self.buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.luts: Dict[str, DensePrototypeLUT] = {}
        self.orig_moments: Dict[str, TensorMoment] = defaultdict(TensorMoment)
        self.lut_moments: Dict[str, TensorMoment] = defaultdict(TensorMoment)
        self.device_lut_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
        self.metrics = LocalMeters()
        self.clean_spike_rates: Dict[str, torch.Tensor] = {}
        self.spike_mse = ScalarMeter()
        self.spike_rate_mse = ScalarMeter()
        self.spike_rate_losses: List[torch.Tensor] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.ssa_names: List[str] = []
        self._register()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset_metrics(self) -> None:
        self.metrics = LocalMeters()
        self.spike_mse = ScalarMeter()
        self.spike_rate_mse = ScalarMeter()
        self.spike_rate_losses = []

    def finalize(self) -> None:
        for lut in self.luts.values():
            lut.finalize()
        self.device_lut_cache.clear()

    def supported_scalar_entries(self) -> int:
        return sum(lut.supported_entries for lut in self.luts.values())

    def memory_kib(self) -> float:
        return self.supported_scalar_entries() * 4.0 / 1024.0

    def set_passthrough(self, collect_clean_spikes: bool = False) -> None:
        self.enabled = False
        self.collect_calibration = False
        self.collect_lut_moments = False
        self.collect_clean_spikes = bool(collect_clean_spikes)
        self.alpha = 0.0
        self.moment_match = False

    def set_eval(self, alpha: float, moment_match: bool = False) -> None:
        self.enabled = True
        self.collect_calibration = False
        self.collect_lut_moments = False
        self.collect_clean_spikes = False
        self.alpha = float(alpha)
        self.moment_match = bool(moment_match)
        self.reset_metrics()

    def _register(self) -> None:
        for name, module in self.model.named_modules():
            if module.__class__.__name__ != "SSA":
                continue
            self.ssa_names.append(name)
            self.handles.append(module.q_lif.register_forward_hook(self._make_store_hook(name, "q")))
            self.handles.append(module.k_lif.register_forward_hook(self._make_store_hook(name, "k")))
            self.handles.append(module.proj_linear.register_forward_hook(self._make_proj_hook(name)))
            self.handles.append(module.proj_lif.register_forward_hook(self._make_proj_lif_hook(name)))

    def _make_store_hook(self, prefix: str, kind: str):
        def hook(_module, _inputs, output):
            self.buffers[prefix][kind] = output.detach()
            if kind in {"q", "k"} and (self.enabled or self.collect_calibration):
                z = output.detach() <= 0
                self.metrics.spike_zero_count += int(z.sum().item())
                self.metrics.spike_value_count += int(z.numel())
            return None

        return hook

    def _make_proj_hook(self, prefix: str):
        def hook(_module, _inputs, output):
            return self._consume_current(prefix, output)

        return hook

    def _make_proj_lif_hook(self, prefix: str):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return None
            y = output.float()
            if self.collect_clean_spikes:
                self.clean_spike_rates[prefix] = y.detach().mean()
            elif self.enabled and prefix in self.clean_spike_rates:
                target = self.clean_spike_rates[prefix].to(device=y.device, dtype=y.dtype)
                rate = y.mean()
                loss = (rate - target).pow(2)
                self.spike_rate_losses.append(loss)
                self.spike_rate_mse.update(float(loss.detach().item()), int(y.numel()))
            return None

        return hook

    def _address(self, prefix: str, current: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor, Tuple[int, ...]]:
        buf = self.buffers[prefix]
        if "q" not in buf or "k" not in buf:
            raise RuntimeError(f"missing q/k buffer for {prefix}")
        module = self.model.get_submodule(prefix)
        q = buf["q"]
        k = buf["k"]
        current_tbnc, shape = to_tbnc_from_proj_linear(current, q)
        address, coarse, address_space, coarse_space = spikformer_qk_scalar_address(
            q_tbnc=q,
            k_tbnc=k,
            num_heads=int(getattr(module, "num_heads")),
            token_bins=self.token_bins,
            channel_bins=self.channel_bins,
            population_bins=self.population_bins,
        )
        if self.mode == "token_channel_mean":
            address = coarse
            address_space = coarse_space
        return address, address_space, current_tbnc, shape

    def _consume_current(self, prefix: str, output: torch.Tensor) -> Optional[torch.Tensor]:
        if "q" not in self.buffers[prefix] or "k" not in self.buffers[prefix]:
            return None
        address, address_space, current_tbnc, shape = self._address(prefix, output)
        lut = self.luts.get(prefix)
        if lut is None:
            lut = DensePrototypeLUT(address_space, mode=self.mode, min_support=self.min_support, seed=self.seed)
            self.luts[prefix] = lut
        if self.collect_calibration:
            lut.update(address, current_tbnc)
            self.orig_moments[prefix].update_tbnc(current_tbnc)
            return None
        if self.collect_lut_moments:
            pred, _seen = self._predict(prefix, address)
            self.lut_moments[prefix].update_tbnc(pred.reshape_as(current_tbnc))
            return None
        if not self.enabled:
            return None
        pred, seen = self._predict(prefix, address)
        lut_tbnc = pred.reshape_as(current_tbnc).to(dtype=current_tbnc.dtype)
        if self.moment_match:
            lut_tbnc = self._moment_match(prefix, lut_tbnc)
        if self.gate_bank is not None:
            lut_tbnc = self.gate_bank(prefix, lut_tbnc)
        blended = current_tbnc + self.alpha * (lut_tbnc - current_tbnc)
        self._update_metrics(current_tbnc, lut_tbnc, blended, seen)
        return restore_proj_linear(blended, shape).to(output.dtype)

    def _predict(self, prefix: str, address: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lut = self.luts[prefix]
        if lut.mean is None or lut.support is None:
            raise RuntimeError("call finalize first")
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
        pred = torch.where(seen, pred, torch.full_like(pred, float(cache["global_mean"].item())))
        return pred, seen

    def _moment_match(self, prefix: str, lut_tbnc: torch.Tensor) -> torch.Tensor:
        if prefix not in self.orig_moments or prefix not in self.lut_moments:
            return lut_tbnc
        om, os = self.orig_moments[prefix].mean_std(lut_tbnc.device, lut_tbnc.dtype)
        lm, ls = self.lut_moments[prefix].mean_std(lut_tbnc.device, lut_tbnc.dtype)
        shape = (om.shape[0], 1, 1, om.shape[1])
        return (lut_tbnc - lm.reshape(shape)) / (ls.reshape(shape) + 1.0e-6) * os.reshape(shape) + om.reshape(shape)

    def _update_metrics(self, orig: torch.Tensor, lut: torch.Tensor, blended: torch.Tensor, seen: torch.Tensor) -> None:
        n = int(orig.numel())
        of = orig.float()
        lf = lut.float()
        bf = blended.float()
        self.metrics.local_mse.update(float(F.mse_loss(lf, of).detach().item()), n)
        self.metrics.local_cosine.update(float(F.cosine_similarity(of.reshape(1, -1), lf.reshape(1, -1), dim=1).detach().item()), n)
        orms = torch.sqrt(torch.mean(of * of) + 1.0e-12)
        lrms = torch.sqrt(torch.mean(lf * lf) + 1.0e-12)
        ostd = of.std(unbiased=False)
        lstd = lf.std(unbiased=False)
        self.metrics.orig_rms.update(float(orms.item()), n)
        self.metrics.lut_rms.update(float(lrms.item()), n)
        self.metrics.rms_ratio.update(float((lrms / orms.clamp_min(1.0e-12)).item()), n)
        self.metrics.orig_std.update(float(ostd.item()), n)
        self.metrics.lut_std.update(float(lstd.item()), n)
        self.metrics.std_ratio.update(float((lstd / ostd.clamp_min(1.0e-12)).item()), n)
        self.metrics.output_zero_frac.update(float((bf.abs() < 1.0e-8).float().mean().item()), n)
        hit = seen.detach().bool()
        self.metrics.hit_count += int(hit.sum().item())
        self.metrics.lookup_count += int(hit.numel())


def flatten_row(row: Dict[str, object]) -> Dict[str, object]:
    probe = row.get("probe", {})
    clean = probe.get("clean", {}) if isinstance(probe, dict) else {}
    injected = probe.get("injected", {}) if isinstance(probe, dict) else {}
    return {
        "method": row.get("method"),
        "inject_pos": "proj_linear_output_before_proj_bn",
        "alpha": row.get("alpha"),
        "trainable_params": row.get("trainable_params", 0),
        "clean_top1": row.get("clean_top1"),
        "injected_top1": row.get("injected_top1"),
        "drop_top1": row.get("drop_top1"),
        "kl_clean_to_injected": row.get("kl_clean_to_injected"),
        "logit_mse": row.get("logit_mse"),
        "current_mse": row.get("local_mse"),
        "current_cosine": row.get("local_cosine"),
        "current_rms_ratio": row.get("orig_lut_rms_ratio"),
        "current_std_ratio": row.get("orig_lut_std_ratio"),
        "hit_rate": row.get("lut_coverage_hit_rate"),
        "fallback_rate": row.get("fallback_rate"),
        "memory_kib": row.get("analytical_memory_footprint_kib"),
        "clean_bn_z": clean.get("bn_mean_abs_z_shift"),
        "injected_bn_z": injected.get("bn_mean_abs_z_shift"),
        "clean_spike_rate": clean.get("lif_spike_rate"),
        "injected_spike_rate": injected.get("lif_spike_rate"),
        "proj_spike_rate_mse": row.get("proj_spike_rate_mse"),
    }


def append_row(result_dir: Path, flat: Dict[str, object], full: Dict[str, object]) -> None:
    path = result_dir / "summary_partial.csv"
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)
    with (result_dir / "rows_partial.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(full, sort_keys=True) + "\n")


@torch.no_grad()
def calibrate(model, loader, device, hook: CurrentLUTHook, batches: int) -> int:
    hook.collect_calibration = True
    hook.enabled = False
    model.eval()
    seen = 0
    for images, _ in loader:
        if seen >= batches:
            break
        reset_snn(model)
        model(images.to(device, non_blocking=True))
        seen += 1
    reset_snn(model)
    hook.collect_calibration = False
    hook.finalize()
    return seen


@torch.no_grad()
def collect_moments(model, loader, device, hook: CurrentLUTHook, batches: int) -> int:
    hook.collect_lut_moments = True
    hook.enabled = False
    seen = 0
    for images, _ in loader:
        if seen >= batches:
            break
        reset_snn(model)
        model(images.to(device, non_blocking=True))
        seen += 1
    reset_snn(model)
    hook.collect_lut_moments = False
    return seen


def evaluate(model, loader, device, hook: CurrentLUTHook, probe: DownstreamProbe, method: str, alpha: float, max_batches: Optional[int], moment_match: bool = False, trainable_params: int = 0) -> Dict[str, object]:
    model.eval()
    probe.reset()
    hook.reset_metrics()
    clean_correct = injected_correct = total = 0
    kl_m = ScalarMeter()
    mse_m = ScalarMeter()
    with torch.no_grad():
        for idx, (images, target) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            hook.set_passthrough(collect_clean_spikes=True)
            probe.set_context("clean")
            reset_snn(model)
            clean_logits = model(images)
            hook.set_eval(alpha=alpha, moment_match=moment_match)
            probe.set_context("injected")
            reset_snn(model)
            injected_logits = model(images)
            clean_correct += int((clean_logits.argmax(1) == target).sum().item())
            injected_correct += int((injected_logits.argmax(1) == target).sum().item())
            total += int(target.numel())
            clean_logp = F.log_softmax(clean_logits.float(), dim=1)
            inj_logp = F.log_softmax(injected_logits.float(), dim=1)
            kl_m.update(float(F.kl_div(inj_logp, clean_logp.exp(), reduction="batchmean").item()), int(target.numel()))
            mse_m.update(float(F.mse_loss(injected_logits.float(), clean_logits.float()).item()), int(target.numel()))
    reset_snn(model)
    local = hook.metrics.summary()
    row = {
        "method": method,
        "alpha": float(alpha),
        "moment_match": bool(moment_match),
        "trainable_params": int(trainable_params),
        "clean_top1": 100.0 * clean_correct / max(total, 1),
        "injected_top1": 100.0 * injected_correct / max(total, 1),
        "drop_top1": 100.0 * (clean_correct - injected_correct) / max(total, 1),
        "kl_clean_to_injected": kl_m.mean(),
        "logit_mse": mse_m.mean(),
        **local,
        "proj_spike_rate_mse": hook.spike_rate_mse.mean(),
        "analytical_memory_footprint_kib": hook.memory_kib(),
        "supported_scalar_entries": hook.supported_scalar_entries(),
        "probe": probe.summary(),
    }
    return row


def train_gate(model, train_loader, device, hook: CurrentLUTHook, gate: GateBank, batches: int, lr: float, lambda_kl: float, lambda_sr: float) -> Dict[str, float]:
    model.eval()
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=0.0)
    total_loss = ScalarMeter()
    total_ce = ScalarMeter()
    total_kl = ScalarMeter()
    total_sr = ScalarMeter()
    seen = 0
    for images, target in train_loader:
        if seen >= batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            hook.set_passthrough(collect_clean_spikes=True)
            reset_snn(model)
            clean_logits = model(images)
        hook.set_eval(alpha=1.0, moment_match=False)
        reset_snn(model)
        opt.zero_grad(set_to_none=True)
        injected_logits = model(images)
        ce = F.cross_entropy(injected_logits, target)
        kl = F.kl_div(F.log_softmax(injected_logits.float(), dim=1), F.softmax(clean_logits.float(), dim=1), reduction="batchmean")
        sr = torch.stack(hook.spike_rate_losses).mean() if hook.spike_rate_losses else torch.zeros((), device=device)
        loss = ce + lambda_kl * kl + lambda_sr * sr
        loss.backward()
        opt.step()
        n = int(target.numel())
        total_loss.update(float(loss.detach().item()), n)
        total_ce.update(float(ce.detach().item()), n)
        total_kl.update(float(kl.detach().item()), n)
        total_sr.update(float(sr.detach().item()), n)
        seen += 1
    reset_snn(model)
    return {"train_batches": seen, "loss": total_loss.mean(), "ce": total_ce.mean(), "kl": total_kl.mean(), "spike_rate_loss": total_sr.mean()}


def make_hooked_model(args, device, gate_variant: str = "none", static_gate: Optional[torch.Tensor] = None):
    model, load_report = build_spikformer(Path(args.repo_root).resolve(), 10, Path(args.checkpoint).resolve())
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    names = [name for name, module in model.named_modules() if module.__class__.__name__ == "SSA"]
    gate = GateBank(names, dim=384, variant=gate_variant, init_temporal=static_gate).to(device) if gate_variant != "none" else None
    hook = CurrentLUTHook(model, mode="aligned_lut", min_support=args.min_support, seed=args.seed, gate_bank=gate)
    probe = DownstreamProbe(model, threshold_eps=args.threshold_eps)
    return model, hook, probe, gate, load_report


def run(args):
    set_seed(args.seed)
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.local_cuda_index}" if torch.cuda.is_available() else "cpu")
    train_loader = cifar_loader(Path(args.data_root).resolve(), "train", args.batch_size, args.workers)
    val_loader = cifar_loader(Path(args.data_root).resolve(), "validation", args.batch_size, args.workers)
    rows: List[Dict[str, object]] = []

    model, hook, probe, gate, load_report = make_hooked_model(args, device)
    calib_batches = calibrate(model, train_loader, device, hook, args.calib_batches)
    moment_batches = collect_moments(model, train_loader, device, hook, args.calib_batches)
    alpha_values = [] if args.alphas.lower() in {"none", "skip", "false"} else [float(x) for x in args.alphas.split(",") if x]
    for alpha in alpha_values:
        row = evaluate(model, val_loader, device, hook, probe, f"current_alpha_{alpha:g}", alpha, args.max_eval_batches)
        rows.append(row)
        append_row(result_dir, flatten_row(row), row)
        print(f"[condition] {row['method']} top1={row['injected_top1']:.4f} drop={row['drop_top1']:.4f}", flush=True)
    row = evaluate(model, val_loader, device, hook, probe, "current_moment_matched_hard", 1.0, args.max_eval_batches, moment_match=True)
    rows.append(row)
    append_row(result_dir, flatten_row(row), row)
    print(f"[condition] {row['method']} top1={row['injected_top1']:.4f} drop={row['drop_top1']:.4f}", flush=True)
    hook.close()
    probe.close()

    static_t = torch.tensor([1.0, 0.75, 0.5, 0.25], dtype=torch.float32)
    model, hook, probe, gate, _ = make_hooked_model(args, device, gate_variant="static_t", static_gate=static_t)
    calibrate(model, train_loader, device, hook, args.calib_batches)
    collect_moments(model, train_loader, device, hook, args.calib_batches)
    row = evaluate(model, val_loader, device, hook, probe, "current_static_temporal_gate_hard", 1.0, args.max_eval_batches, trainable_params=0)
    rows.append(row)
    append_row(result_dir, flatten_row(row), row)
    print(f"[condition] {row['method']} top1={row['injected_top1']:.4f} drop={row['drop_top1']:.4f}", flush=True)
    hook.close()
    probe.close()

    for variant, label in [("learned_t", "current_learned_temporal_gate_hard"), ("learned_tc", "current_learned_temporal_channel_gate_hard")]:
        model, hook, probe, gate, _ = make_hooked_model(args, device, gate_variant=variant)
        calibrate(model, train_loader, device, hook, args.calib_batches)
        collect_moments(model, train_loader, device, hook, args.calib_batches)
        train_report = train_gate(model, train_loader, device, hook, gate, args.train_batches, args.lr, args.lambda_kl, args.lambda_sr)
        row = evaluate(model, val_loader, device, hook, probe, label, 1.0, args.max_eval_batches, trainable_params=gate.parameter_count())
        row["train_report"] = train_report
        rows.append(row)
        append_row(result_dir, flatten_row(row), row)
        print(f"[condition] {row['method']} top1={row['injected_top1']:.4f} drop={row['drop_top1']:.4f}", flush=True)
        hook.close()
        probe.close()

    flat_rows = [flatten_row(r) for r in rows]
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    payload = {
        "experiment": "spikformer_current_level_lut_injection",
        "claim": "diagnostic hook simulation of replacing learned current response; does not remove dense proj_linear compute",
        "python": os.environ.get("PYTHON", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "repo_root": str(Path(args.repo_root).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "load_report": load_report,
        "calib_batches": calib_batches,
        "moment_batches": moment_batches,
        "args": vars(args),
        "rows": rows,
    }
    write_json(result_dir / "metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--train-batches", type=int, default=196)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-cuda-index", type=int, default=0)
    parser.add_argument("--alphas", default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-sr", type=float, default=10.0)
    parser.add_argument("--threshold-eps", type=float, default=0.05)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"result_dir": str(Path(args.result_dir).resolve()), "rows": len(payload["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
