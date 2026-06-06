#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
        mode_seed: int,
    ) -> None:
        super().__init__()
        self.name = proto.name
        self.kind = proto.kind
        self.stage = proto.stage
        self.mode = str(mode)
        self.address_space = int(proto.address_space)
        self.mode_seed = int(mode_seed)
        if self.mode not in {"address_lut", "global_mean", "token_channel_lut", "shuffled_address_lut"}:
            raise ValueError(f"unsupported E3 adapter mode: {self.mode}")

        if self.mode == "global_mean":
            init = torch.tensor([float(proto.global_mean)], dtype=torch.float32)
        elif self.mode == "token_channel_lut":
            init = self._coarse_init(proto, shrinkage_tau)
        else:
            init = self._address_init(proto, shrinkage_tau)
        self.table = nn.Parameter(init)

        alpha = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        alpha_logit = math.log(alpha / (1.0 - alpha))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32), requires_grad=bool(learn_alpha))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, address: torch.Tensor, response: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        index = self._index(address, response.device)
        pred = self.table.to(device=response.device, dtype=response.dtype)[index]
        return response + self.alpha.to(device=response.device, dtype=response.dtype) * (pred - response), index

    def _index(self, address: torch.Tensor, device: torch.device) -> torch.Tensor:
        addr = address.detach().to(device=device, dtype=torch.long)
        if self.mode == "global_mean":
            return torch.zeros_like(addr)
        if self.mode == "token_channel_lut":
            return torch.div(addr, 4, rounding_mode="floor").clamp_max(self.table.numel() - 1)
        if self.mode == "shuffled_address_lut":
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

    def _address_init(self, proto: ModulePrototype, shrinkage_tau: float) -> torch.Tensor:
        means = proto.address_mean.to(torch.float32)
        counts = proto.count.to(torch.float32)
        global_mean = torch.full_like(means, fill_value=float(proto.global_mean))
        if shrinkage_tau <= 0:
            return means
        weight = counts / (counts + float(shrinkage_tau))
        return weight * means + (1.0 - weight) * global_mean

    def _coarse_init(self, proto: ModulePrototype, shrinkage_tau: float) -> torch.Tensor:
        coarse_space = max(1, proto.address_space // 4)
        coarse_sum = torch.zeros(coarse_space, dtype=torch.float64)
        coarse_count = torch.zeros(coarse_space, dtype=torch.float64)
        coarse_index = torch.arange(proto.address_space, dtype=torch.long) // 4
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

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "stage": self.stage,
            "mode": self.mode,
            "address_space": self.address_space,
            "table_entries": int(self.table.numel()),
            "trainable_parameters": sum(param.numel() for param in self.parameters() if param.requires_grad),
            "alpha": float(self.alpha.detach().cpu().item()),
        }


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
        mode_seed: int,
    ) -> None:
        super().__init__()
        self.model = model
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
                    mode_seed=mode_seed,
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
    env_calib_batches = os.environ.get("QKFORMER_LUT_E3_CALIB_BATCHES")
    if env_calib_batches:
        data_cfg["calibration"]["num_batches"] = int(env_calib_batches)
    env_train_batches = os.environ.get("QKFORMER_LUT_E3_TRAIN_BATCHES")
    if env_train_batches:
        data_cfg["train"]["num_batches"] = int(env_train_batches)
    env_eval_batches = os.environ.get("QKFORMER_LUT_E3_EVAL_BATCHES")
    if env_eval_batches:
        data_cfg["evaluation"]["num_batches"] = int(env_eval_batches)
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
        "QKFORMER_LUT_E3_CALIB_BATCHES": env_calib_batches,
        "QKFORMER_LUT_E3_TRAIN_BATCHES": env_train_batches,
        "QKFORMER_LUT_E3_EVAL_BATCHES": env_eval_batches,
    }


def train_one_epoch(model, adapter, loader, data_cfg, train_cfg, optimizer, device) -> Dict[str, float]:
    model.eval()
    adapter.train()
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


def evaluate(model, adapter, loader, data_cfg, device) -> Dict[str, object]:
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
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = int(targets.numel())
            adapter.enabled = False
            reset_model_state(model)
            baseline_logits = model(images)
            b_loss = loss_fn(baseline_logits, targets)
            b_top1, b_top5 = accuracy(baseline_logits, targets)
            reset_model_state(model)

            adapter.clear_step()
            adapter.enabled = True
            reset_model_state(model)
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
    adapter.enabled = False
    return {
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


def run(config_path: Path, output_dir: Path) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(config_path)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    model_cfg = dict(cfg["model"])
    data_cfg = {key: dict(value) for key, value in dict(cfg["data"]).items()}
    diag_cfg = dict(cfg["diagnostic"])
    adapter_cfg = dict(cfg["adapter"])
    train_cfg = dict(cfg["train"])
    env_overrides = apply_env_overrides(model_cfg, data_cfg, adapter_cfg, train_cfg)

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

    print("[qk-lut-e3] evaluation_start")
    evaluation = evaluate(model, adapter, evaluation_loader, data_cfg["evaluation"], device)
    adapter.close()
    metrics = {
        "experiment": cfg["experiment"],
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
        "env_overrides": env_overrides,
        "adapter_summary": adapter.summary(),
        "calibration_prototypes": bank.summary(),
        "calibration_hook_summary": calibration_hook_summary,
        "train_history": train_history,
        "classification": evaluation,
        "verdict": "PENDING_PHASE_GATE_REVIEW" if checkpoint_info["loaded"] else "PENDING_REAL_DATA_CHECKPOINT",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
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
