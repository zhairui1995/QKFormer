#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from qkformer_lut.cross_arch_hooks import SEWResNetBasicBlockLUTHook, SpikformerSSALUTHook


MODES = ("aligned_lut", "shuffled_address", "token_channel_mean")


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


def cifar_loader(data_root: Path, dataset: str, split: str, batch_size: int, workers: int):
    ds_cls = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    ds = ds_cls(
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


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, max_batches: Optional[int] = None) -> Tuple[float, int]:
    model.eval()
    correct = 0
    total = 0
    for idx, (images, target) in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        reset_snn(model)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += int((pred == target).sum().item())
        total += int(target.numel())
    reset_snn(model)
    return 100.0 * correct / max(total, 1), total


@torch.no_grad()
def calibrate(model: nn.Module, loader, device: torch.device, hook, max_batches: int) -> int:
    model.eval()
    batches = 0
    hook.replace = False
    for images, _target in loader:
        if batches >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        reset_snn(model)
        _ = model(images)
        batches += 1
    reset_snn(model)
    hook.finalize()
    return batches


def build_spikformer(repo: Path, num_classes: int, checkpoint: Optional[Path]) -> Tuple[nn.Module, Dict[str, object]]:
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError("Spikformer Track A requires an explicit CIFAR-10 checkpoint; none was found.")
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


def build_cifar_sew_resnet34(repo: Path, num_classes: int, checkpoint: Optional[Path]) -> Tuple[nn.Module, Dict[str, object]]:
    cext_fallback = False
    try:
        from spikingjelly.cext import neuron as _cext_neuron  # noqa: F401
    except Exception:
        from spikingjelly.clock_driven import neuron as py_neuron

        cext_pkg = types.ModuleType("spikingjelly.cext")
        cext_neuron = types.ModuleType("spikingjelly.cext.neuron")
        cext_neuron.MultiStepIFNode = py_neuron.MultiStepIFNode
        cext_neuron.MultiStepLIFNode = py_neuron.MultiStepLIFNode
        sys.modules["spikingjelly.cext"] = cext_pkg
        sys.modules["spikingjelly.cext.neuron"] = cext_neuron
        cext_pkg.neuron = cext_neuron
        cext_fallback = True

    sys.path.insert(0, str(repo / "imagenet"))
    try:
        sew_resnet = importlib.import_module("sew_resnet")
        model = sew_resnet.sew_resnet34(num_classes=num_classes, T=4, connect_f="ADD")
    finally:
        try:
            sys.path.remove(str(repo / "imagenet"))
        except ValueError:
            pass

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)

    report: Dict[str, object] = {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "loaded_keys": 0,
        "skipped_keys": [],
        "cext_neuron_fallback_to_python": cext_fallback,
    }
    if checkpoint is None or not checkpoint.exists():
        report["warning"] = "no ANN ResNet-34 checkpoint found; SEW model is random-init except CIFAR stem/head"
        return model, report

    src = load_state_dict(checkpoint)
    dst = model.state_dict()
    direct = {key: value for key, value in src.items() if key in dst and tuple(value.shape) == tuple(dst[key].shape)}
    direct_fraction = len(direct) / max(len(dst), 1)
    if direct_fraction > 0.5:
        msg = model.load_state_dict(direct, strict=False)
        report.update(
            {
                "load_mode": "direct_trained_sew_checkpoint",
                "loaded_keys": len(direct),
                "missing_after_direct_load": list(msg.missing_keys)[:50],
                "unexpected_after_direct_load": list(msg.unexpected_keys),
            }
        )
        return model, report

    mapped: Dict[str, torch.Tensor] = {}
    skipped = []

    def add(dst_key: str, src_key: str) -> None:
        if src_key in src and dst_key in dst and tuple(src[src_key].shape) == tuple(dst[dst_key].shape):
            mapped[dst_key] = src[src_key]
        elif src_key in src:
            skipped.append({"dst": dst_key, "src": src_key, "src_shape": list(src[src_key].shape), "dst_shape": list(dst[dst_key].shape) if dst_key in dst else None})

    for key in ("bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked"):
        add(key, key)
    for layer in range(1, 5):
        blocks = 3 if layer == 1 else 4 if layer == 2 else 6 if layer == 3 else 3
        for block in range(blocks):
            prefix = f"layer{layer}.{block}"
            for conv in ("conv1", "conv2"):
                add(f"{prefix}.{conv}.module.0.weight", f"{prefix}.{conv}.weight")
                bn = "bn1" if conv == "conv1" else "bn2"
                for suf in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    add(f"{prefix}.{conv}.module.1.{suf}", f"{prefix}.{bn}.{suf}")
            for suf in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                add(f"{prefix}.downsample.0.module.1.{suf}", f"{prefix}.downsample.1.{suf}")
            add(f"{prefix}.downsample.0.module.0.weight", f"{prefix}.downsample.0.weight")

    msg = model.load_state_dict(mapped, strict=False)
    report.update(
        {
            "load_mode": "partial_ann_resnet34_mapping",
            "loaded_keys": len(mapped),
            "skipped_keys": skipped[:20],
            "missing_after_partial_load": list(msg.missing_keys)[:50],
            "unexpected_after_partial_load": list(msg.unexpected_keys),
        }
    )
    return model, report


def run_track(args) -> Dict[str, object]:
    set_seed(args.seed)
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.local_cuda_index}" if torch.cuda.is_available() else "cpu")
    num_classes = 100 if args.dataset == "cifar100" else 10
    data_root = Path(args.data_root).resolve()
    calib_loader = cifar_loader(data_root, args.dataset, "train", args.batch_size, args.workers)
    val_loader = cifar_loader(data_root, args.dataset, "validation", args.batch_size, args.workers)

    if args.track == "spikformer":
        model, load_report = build_spikformer(Path(args.repo_root).resolve(), num_classes, Path(args.checkpoint).resolve() if args.checkpoint else None)
        hook_cls = SpikformerSSALUTHook
    elif args.track == "sew_resnet34":
        model, load_report = build_cifar_sew_resnet34(Path(args.repo_root).resolve(), num_classes, Path(args.checkpoint).resolve() if args.checkpoint else None)
        hook_cls = SEWResNetBasicBlockLUTHook
    else:
        raise ValueError(args.track)

    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    baseline_top1, baseline_n = evaluate(model, val_loader, device, args.max_eval_batches)
    rows = []
    for mode in MODES:
        hook = hook_cls(model, mode=mode, min_support=args.min_support, replace=False, seed=args.seed)
        calib_batches = calibrate(model, calib_loader, device, hook, args.calib_batches)
        hook.replace = True
        top1, n = evaluate(model, val_loader, device, args.max_eval_batches)
        metrics = hook.summary()
        hook.close()
        rows.append(
            {
                "mode": mode,
                "baseline_top1": baseline_top1,
                "replacement_top1": top1,
                "delta_top1": top1 - baseline_top1,
                "drop_top1": baseline_top1 - top1,
                "num_eval_samples": n,
                "calib_batches": calib_batches,
                "hook_metrics": metrics,
                "gate": "PASS" if baseline_top1 - top1 < args.max_drop_pct else "STOP",
            }
        )

    verdict = "PASS" if all(row["drop_top1"] < args.max_drop_pct for row in rows) else "STOP"
    payload = {
        "track": args.track,
        "dataset": args.dataset,
        "phase": "phase1_cifar10_pilot" if args.dataset == "cifar10" else "phase2_full",
        "verdict": verdict,
        "success_gate": f"Acc@1 drop < {args.max_drop_pct} percentage points for every LUT/control mode",
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "repo_root": str(Path(args.repo_root).resolve()),
        "data_root": str(data_root),
        "load_report": load_report,
        "baseline": {"top1": baseline_top1, "num_eval_samples": baseline_n},
        "rows": rows,
    }
    write_json(result_dir / "metrics.json", payload)
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "baseline_top1", "replacement_top1", "delta_top1", "drop_top1", "gate"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("spikformer", "sew_resnet34"), required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-drop-pct", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-cuda-index", type=int, default=0)
    args = parser.parse_args()

    try:
        payload = run_track(args)
        print(json.dumps({"result_dir": str(Path(args.result_dir).resolve()), "verdict": payload["verdict"]}, sort_keys=True))
    except Exception as exc:
        result_dir = Path(args.result_dir).resolve()
        write_json(
            result_dir / "metrics.json",
            {
                "track": args.track,
                "dataset": args.dataset,
                "verdict": "BLOCKED",
                "error": repr(exc),
                "python": sys.executable,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "repo_root": str(Path(args.repo_root).resolve()),
                "data_root": str(Path(args.data_root).resolve()),
            },
        )
        raise


if __name__ == "__main__":
    main()
