#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import yaml

from qkformer_lut.hooks import QKAddressDiagnostic


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cifar10_model(root: Path, cfg: Dict[str, object]) -> torch.nn.Module:
    model_dir = root / "cifar10"
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
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    return loader, "cifar10"


def build_loader(root: Path, cfg: Dict[str, object], model_cfg: Dict[str, object], device: torch.device):
    mode = str(cfg.get("mode", "auto"))
    batch_size = int(cfg["batch_size"])
    if mode in {"auto", "cifar10"}:
        try:
            return make_cifar10_loader(
                data_dir=str(cfg["data_dir"]),
                split=str(cfg.get("split", "validation")),
                batch_size=batch_size,
                workers=int(cfg.get("workers", 4)),
                device=device,
            )
        except Exception as exc:
            if mode == "cifar10":
                raise
            print(f"[qk-lut-e0] CIFAR-10 loader unavailable, using synthetic input: {exc}")
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


def run(config_path: Path, output_dir: Path) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(config_path)
    set_seed(int(cfg["experiment"].get("seed", 42)))

    model_cfg = dict(cfg["model"])
    diag_cfg = dict(cfg["diagnostic"])
    data_cfg = dict(cfg["data"])
    device_name = str(diag_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the upstream cupy-backed QKFormer LIF modules")
    device = torch.device(device_name)

    if model_cfg.get("family") != "cifar10":
        raise ValueError("E0 currently supports model.family=cifar10")
    env_checkpoint = os.environ.get("QKFORMER_LUT_CKPT")
    if env_checkpoint:
        model_cfg["checkpoint"] = env_checkpoint

    model = build_cifar10_model(root, model_cfg)
    checkpoint_info = load_checkpoint_if_available(model, model_cfg.get("checkpoint"))
    model.to(device)
    model.eval()

    loader, data_source = build_loader(root, data_cfg, model_cfg, device)
    diagnostic = QKAddressDiagnostic(
        model,
        token_bins=int(diag_cfg.get("token_bins", 8)),
        channel_bins=int(diag_cfg.get("channel_bins", 8)),
        population_bins=int(diag_cfg.get("population_bins", 4)),
        max_records_per_module_per_batch=int(diag_cfg.get("max_records_per_module_per_batch", 65536)),
    )

    max_batches = int(data_cfg["num_batches"])
    processed = 0
    with torch.no_grad():
        for images, _targets in loader:
            if processed >= max_batches:
                break
            if not torch.is_tensor(images):
                images = images[0]
            images = images.to(device, non_blocking=True)
            _ = model(images)
            processed += 1

    summary = diagnostic.summary()
    diagnostic.close()
    verdict = "PENDING"
    if data_source == "synthetic" or not checkpoint_info["loaded"]:
        verdict = "PENDING_REAL_DATA_CHECKPOINT"

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
            "source": data_source,
            "num_batches": processed,
            "batch_size": data_cfg["batch_size"],
        },
        "diagnostic_config": diag_cfg,
        "verdict": verdict,
        **summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"[qk-lut-e0] wrote {metrics_path}")
    print(f"[qk-lut-e0] verdict={verdict}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QK-LUTFormer E0 diagnostic")
    parser.add_argument("--config", type=Path, default=Path("configs/qkformer_lut_e0_diag.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
