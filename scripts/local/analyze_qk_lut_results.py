#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: (path.stat().st_mtime, str(path)))


def latest_t1_train(results: Path) -> Path | None:
    candidates: list[Path] = []
    for path in results.glob("qkformer_cifar10_train_*"):
        manifest = path / "checkpoint_manifest.json"
        if not manifest.exists():
            continue
        data = load_json(manifest)
        if str(data.get("time_step")) == "1":
            candidates.append(path)
    return latest(candidates)


def latest_t1_e0(results: Path) -> Path | None:
    candidates: list[Path] = []
    for metrics in results.glob("qkformer_lut_e0_diag_*/metrics.json"):
        data = load_json(metrics)
        if str(data.get("model", {}).get("time_step")) == "1":
            candidates.append(metrics.parent)
    return latest(candidates)


def latest_t1_e3(results: Path, limit: int) -> list[Path]:
    candidates: list[Path] = []
    for metrics in results.glob("qkformer_lut_e3_trainable_lut_*/metrics.json"):
        data = load_json(metrics)
        if str(data.get("model", {}).get("time_step")) == "1":
            candidates.append(metrics.parent)
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)[:limit]


def best_summary_row(train_dir: Path) -> tuple[str | None, str | None, str | None]:
    summaries = list(train_dir.glob("**/summary.csv"))
    if not summaries:
        return None, None, None
    rows = list(csv.DictReader(summaries[0].open(newline="", encoding="utf-8")))
    if not rows:
        return None, None, None
    best = max(rows, key=lambda row: float(row.get("eval_top1") or -1))
    last = rows[-1]
    return best.get("epoch"), best.get("eval_top1"), last.get("eval_top1")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize QK-LUTFormer local results.")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--e3-count", type=int, default=9, help="Latest T=1 E3 runs to summarize")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    results = root / "results"
    print(f"[qk-lut-analyze] root={root}")

    train_dir = latest_t1_train(results)
    if train_dir:
        epoch, best_top1, last_top1 = best_summary_row(train_dir)
        print(f"[qk-lut-analyze] latest_t1_train={train_dir.name}")
        print(f"[qk-lut-analyze] train_best_epoch={epoch} train_best_top1={best_top1} train_last_top1={last_top1}")
    else:
        print("[qk-lut-analyze] latest_t1_train=missing")

    e0_dir = latest_t1_e0(results)
    if e0_dir:
        e0 = load_json(e0_dir / "metrics.json")
        print(f"[qk-lut-analyze] latest_t1_e0={e0_dir.name}")
        print(
            "[qk-lut-analyze] e0 coverage="
            f"{fmt(e0.get('address_coverage'), 6)} singleton={fmt(e0.get('singleton_fraction'), 6)} "
            f"conditional_variance={fmt(e0.get('conditional_variance'), 6)}"
        )
    else:
        print("[qk-lut-analyze] latest_t1_e0=missing")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run_dir in reversed(latest_t1_e3(results, args.e3_count)):
        metrics = load_json(run_dir / "metrics.json")
        mode = metrics.get("adapter_summary", {}).get("mode") or metrics.get("adapter_config", {}).get("mode", "unknown")
        cls = metrics.get("classification", {})
        replacement = cls.get("replacement", {})
        row = {
            "dir": run_dir.name,
            "seed": metrics.get("experiment", {}).get("seed"),
            "mode": mode,
            "baseline_top1": cls.get("baseline", {}).get("top1"),
            "replacement_top1": replacement.get("top1"),
            "delta_top1": cls.get("delta", {}).get("top1"),
            "baseline_loss": cls.get("baseline", {}).get("loss"),
            "replacement_loss": replacement.get("loss"),
            "delta_loss": cls.get("delta", {}).get("loss"),
            "kl": replacement.get("kl_to_baseline"),
            "logit_mse": replacement.get("logit_mse"),
            "local_mse": replacement.get("local_mse"),
        }
        grouped.setdefault(mode, []).append(row)
        print(
            "[qk-lut-analyze] e3 "
            f"dir={row['dir']} mode={mode} seed={row['seed']} "
            f"top1={fmt(row['baseline_top1'], 2)}->{fmt(row['replacement_top1'], 2)} "
            f"delta={fmt(row['delta_top1'], 4)} loss_delta={fmt(row['delta_loss'], 6)} "
            f"kl={fmt(row['kl'], 6)} logit_mse={fmt(row['logit_mse'], 6)} local_mse={fmt(row['local_mse'], 6)}"
        )

    means: dict[str, float] = {}
    for mode, rows in sorted(grouped.items()):
        deltas = [float(row["delta_top1"]) for row in rows if row["delta_top1"] is not None]
        losses = [float(row["delta_loss"]) for row in rows if row["delta_loss"] is not None]
        if deltas:
            means[mode] = statistics.mean(deltas)
            print(
                "[qk-lut-analyze] summary "
                f"mode={mode} n={len(deltas)} mean_delta_top1={fmt(means[mode], 4)} "
                f"min_delta_top1={fmt(min(deltas), 4)} max_delta_top1={fmt(max(deltas), 4)} "
                f"mean_delta_loss={fmt(statistics.mean(losses) if losses else None, 6)}"
            )

    verdict = "PENDING"
    if {"address_lut", "global_mean", "token_channel_lut"}.issubset(means):
        address = means["address_lut"]
        best_control = max(means["global_mean"], means["token_channel_lut"])
        if address > 0 and address > best_control:
            verdict = "CONDITIONAL GO: address_lut leads T=1 controls; repeat or broaden before paper claim"
        elif address > best_control:
            verdict = "CONDITIONAL GO: address_lut beats controls but mean gain is not positive"
        else:
            verdict = "NO-GO for address-specific accuracy claim at this setting"
    print(f"[qk-lut-analyze] verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
