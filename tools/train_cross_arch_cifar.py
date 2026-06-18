#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from run_cross_arch_lut_phase1 import build_cifar_sew_resnet34, reset_snn


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cifar_loader(data_root: Path, dataset: str, split: str, batch_size: int, workers: int):
    ds_cls = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    train = split == "train"
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    ds = ds_cls(root=str(data_root), train=train, download=False, transform=transform_train if train else transform_eval)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def build_spikformer(repo: Path, num_classes: int) -> Tuple[nn.Module, Dict[str, object]]:
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
    return model, {"init": "random", "arch": "Spikformer-4-384w", "time_step": 4}


def build_model(args, num_classes: int) -> Tuple[nn.Module, Dict[str, object]]:
    if args.track == "spikformer":
        return build_spikformer(Path(args.repo_root).resolve(), num_classes)
    if args.track == "sew_resnet34":
        model, report = build_cifar_sew_resnet34(Path(args.repo_root).resolve(), num_classes, None)
        report["init"] = "random_cifar_stem_head"
        report["arch"] = "SEW-ResNet-34"
        report["time_step"] = 4
        return model, report
    raise ValueError(args.track)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, max_batches: int = 0) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (images, target) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        reset_snn(model)
        logits = model(images)
        loss = loss_fn(logits, target)
        pred = logits.argmax(dim=1)
        total += int(target.numel())
        correct += int((pred == target).sum().item())
        loss_sum += float(loss.item()) * int(target.numel())
    reset_snn(model)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    amp: bool,
    max_batches: int = 0,
) -> Tuple[float, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch_idx, (images, target) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        reset_snn(model)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, target)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        pred = logits.detach().argmax(dim=1)
        total += int(target.numel())
        correct += int((pred == target).sum().item())
        loss_sum += float(loss.detach().item()) * int(target.numel())
    reset_snn(model)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("spikformer", "sew_resnet34"), required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 100 if args.dataset == "cifar100" else 10

    model, init_report = build_model(args, num_classes)
    model.to(device)
    train_loader = cifar_loader(Path(args.data_root).resolve(), args.dataset, "train", args.batch_size, args.workers)
    val_loader = cifar_loader(Path(args.data_root).resolve(), args.dataset, "validation", args.val_batch_size, args.workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    best_top1 = -1.0
    best_epoch = -1
    rows = []
    best_path = result_dir / "model_best.pth.tar"
    latest_path = result_dir / "checkpoint_latest.pth.tar"
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_top1", "val_loss", "val_top1", "lr"])
        writer.writeheader()
        for epoch in range(args.epochs):
            train_loss, train_top1 = train_one_epoch(
                model, train_loader, optimizer, scaler, device, args.amp, args.max_train_batches
            )
            val_loss, val_top1 = evaluate(model, val_loader, device, args.max_val_batches)
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "lr": optimizer.param_groups[0]["lr"],
            }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_top1": max(best_top1, val_top1),
                "best_epoch": best_epoch if best_top1 >= val_top1 else epoch,
                "args": vars(args),
            }
            torch.save(payload, latest_path)
            if val_top1 > best_top1:
                best_top1 = val_top1
                best_epoch = epoch
                torch.save(payload, best_path)
            print(
                f"[cross-train] track={args.track} dataset={args.dataset} epoch={epoch} "
                f"train_top1={train_top1:.4f} val_top1={val_top1:.4f} best={best_top1:.4f}@{best_epoch}",
                flush=True,
            )

    manifest = {
        "track": args.track,
        "dataset": args.dataset,
        "result_dir": str(result_dir),
        "python": sys.executable,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "init_report": init_report,
        "best_top1": best_top1,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "epochs": args.epochs,
        "rows": rows,
    }
    (result_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": str(best_path), "best_top1": best_top1, "result_dir": str(result_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
