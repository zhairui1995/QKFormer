#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _best_train_dir(results_root: Path) -> Optional[Path]:
    dirs = []
    for path in results_root.glob("qkformer_cifar100_train_*"):
        manifest = _load_json(path / "checkpoint_manifest.json")
        if str(manifest.get("time_step")) == "4":
            dirs.append(path)
    if not dirs:
        return None
    return max(dirs, key=lambda p: (p.stat().st_mtime, str(p)))


def _parse_summary_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    best = None
    for row in rows:
        keys = {key.lower(): key for key in row.keys()}
        top1_key = next((key for low, key in keys.items() if low in {"eval_top1", "top1", "acc1", "acc@1"}), None)
        epoch_key = next((key for low, key in keys.items() if low == "epoch"), None)
        if top1_key is None:
            continue
        try:
            top1 = float(row[top1_key])
        except Exception:
            continue
        candidate = {
            "epoch": int(float(row[epoch_key])) if epoch_key and row.get(epoch_key) not in {"", None} else None,
            "top1": top1,
            "row": row,
        }
        if best is None or top1 > best["top1"]:
            best = candidate
    return {"best": best, "num_rows": len(rows)}


def _parse_train_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    pattern = re.compile(r"Best metric:\s*([0-9.]+)\s*\(epoch\s*(\d+)\)")
    best = None
    for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
        candidate = {"top1": float(match.group(1)), "epoch": int(match.group(2))}
        if best is None or candidate["top1"] > best["top1"]:
            best = candidate
    return {"best": best, "source": str(path)} if best else {}


def _checkpoint_path(metrics: Dict[str, Any]) -> str:
    ckpt = metrics.get("model", {}).get("checkpoint", {})
    return str(ckpt.get("path") or "")


def _matches_c100_t4(metrics: Dict[str, Any], train_dir: Optional[Path]) -> bool:
    model = metrics.get("model", {})
    if model.get("family") != "cifar100" or str(model.get("time_step")) != "4":
        return False
    if train_dir is None:
        return True
    checkpoint_parts = Path(_checkpoint_path(metrics)).parts
    return train_dir.name in checkpoint_parts


def _latest_e1(results_root: Path, train_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    candidates = []
    for path in results_root.glob("qkformer_lut_e1_recon_*/metrics.json"):
        metrics = _load_json(path)
        if _matches_c100_t4(metrics, train_dir):
            candidates.append((path.parent.stat().st_mtime, path, metrics))
    if not candidates:
        return None
    _, path, metrics = max(candidates, key=lambda item: (item[0], str(item[1])))
    overall = metrics.get("overall_reconstruction", {})
    return {
        "result_dir": str(path.parent),
        "address_reduction_pct": 100.0 * float(overall.get("address_relative_mse_reduction", 0.0)),
        "token_channel_reduction_pct": 100.0 * float(
            overall.get("token_channel_relative_mse_reduction", 0.0)
        ),
        "shuffled_reduction_pct": 100.0 * float(
            overall.get("shuffled_address_relative_mse_reduction", 0.0)
        ),
        "hierarchy_reduction_pct": 100.0
        * float(overall.get("hierarchical_backoff_relative_mse_reduction", 0.0)),
        "global_mean_mse": overall.get("global_mean_mse"),
        "address_mse": overall.get("address_lut_mse"),
    }


def _e3_rows(results_root: Path, train_dir: Optional[Path]) -> List[Dict[str, Any]]:
    rows = []
    for path in results_root.glob("qkformer_lut_e3_trainable_lut_*/metrics.json"):
        metrics = _load_json(path)
        if not _matches_c100_t4(metrics, train_dir):
            continue
        protocol = metrics.get("protocol", {})
        if int(protocol.get("version", 0)) < 2:
            continue
        cls = metrics.get("classification", {})
        adapter = metrics.get("adapter_summary", {})
        delta = cls.get("delta", {})
        baseline = cls.get("baseline", {})
        replacement = cls.get("replacement", {})
        rows.append(
            {
                "result_dir": str(path.parent),
                "mode": adapter.get("mode"),
                "seed": metrics.get("experiment", {}).get("seed"),
                "target_modules": ",".join(adapter.get("target_modules", []) or []),
                "alpha": adapter.get("alpha"),
                "baseline_top1": baseline.get("top1"),
                "replacement_top1": replacement.get("top1"),
                "delta_top1": delta.get("top1"),
                "baseline_loss": baseline.get("loss"),
                "replacement_loss": replacement.get("loss"),
                "delta_loss": delta.get("loss"),
                "logit_mse": cls.get("logit_mse"),
                "kl_to_baseline": cls.get("kl_to_baseline"),
                "local_mse": metrics.get("local_reconstruction", {}).get("mse"),
                "protocol_version": protocol.get("version"),
                "evaluation_batch_size": protocol.get("evaluation_batch_size"),
            }
        )
    return sorted(rows, key=lambda row: (str(row["mode"]), str(row["seed"]), row["result_dir"]))


def _summarize_e3(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    summary = []
    for mode, items in sorted(grouped.items()):
        valid = [row for row in items if row.get("delta_top1") is not None]
        if not valid:
            continue
        deltas = [float(row["delta_top1"]) for row in valid]
        losses = [float(row["delta_loss"]) for row in valid if row.get("delta_loss") is not None]
        replacements = [
            float(row["replacement_top1"]) for row in valid if row.get("replacement_top1") is not None
        ]
        summary.append(
            {
                "mode": mode,
                "n": len(valid),
                "mean_delta_top1": sum(deltas) / len(deltas),
                "min_delta_top1": min(deltas),
                "max_delta_top1": max(deltas),
                "mean_replacement_top1": sum(replacements) / len(replacements) if replacements else None,
                "mean_delta_loss": sum(losses) / len(losses) if losses else None,
            }
        )
    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# CIFAR-100 T=4 QK-LUTFormer Report",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        f"- Train dir: `{report.get('train_dir')}`",
        f"- Baseline best Acc@1: `{report.get('baseline_best_top1')}`",
        f"- Corrected-protocol baseline spread: `{report['checks'].get('baseline_spread')}`",
        f"- Address adapter beats training best: `{report['checks'].get('address_beats_training_best')}`",
        "",
        "## E1 Reconstruction",
    ]
    e1 = report.get("e1") or {}
    if e1:
        lines.extend(
            [
                f"- Address reduction: `{e1['address_reduction_pct']:.4f}%`",
                f"- Token/channel reduction: `{e1['token_channel_reduction_pct']:.4f}%`",
                f"- Shuffled reduction: `{e1['shuffled_reduction_pct']:.4f}%`",
            ]
        )
    else:
        lines.append("- Missing.")
    lines.extend(
        [
            "",
            "## E3 Adapter Summary",
            "| Mode | n | Mean delta Acc@1 | Min | Max | Mean replacement Acc@1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report.get("e3_summary", []):
        lines.append(
            "| {mode} | {n} | {mean:.4f} | {minv:.4f} | {maxv:.4f} | {repl} |".format(
                mode=row["mode"],
                n=row["n"],
                mean=row["mean_delta_top1"],
                minv=row["min_delta_top1"],
                maxv=row["max_delta_top1"],
                repl=(
                    "-"
                    if row["mean_replacement_top1"] is None
                    else f"{row['mean_replacement_top1']:.4f}"
                ),
            )
        )
    lines.extend(["", "## Interpretation", report["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CIFAR-100 T=4 QK-LUTFormer results")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", type=Path, default=Path("results/qk_lutformer_cifar100_t4"))
    args = parser.parse_args()

    train_dir = _best_train_dir(args.results_root)
    train_manifest = _load_json(train_dir / "checkpoint_manifest.json") if train_dir else {}
    summary_csv_value = str(train_manifest.get("summary_csv") or "") if train_manifest else ""
    summary_csv = Path(summary_csv_value) if summary_csv_value else None
    train_summary = _parse_summary_csv(summary_csv) if summary_csv and summary_csv.is_file() else {}
    if not train_summary:
        train_summary = _parse_train_log(train_dir / "train_log.txt") if train_dir else {}
    baseline_best = (train_summary.get("best") or {}).get("top1")

    e1 = _latest_e1(args.results_root, train_dir)
    e3_rows = _e3_rows(args.results_root, train_dir)
    e3_summary = _summarize_e3(e3_rows)
    by_mode = {row["mode"]: row for row in e3_summary}
    address = by_mode.get("address_lut")
    controls = [by_mode.get(name) for name in ("global_mean", "token_channel_lut", "shuffled_address_lut")]
    controls = [row for row in controls if row is not None]

    baseline_values = [
        float(row["baseline_top1"])
        for row in e3_rows
        if row.get("baseline_top1") is not None
    ]
    baseline_spread = max(baseline_values) - min(baseline_values) if baseline_values else None
    protocol_valid = bool(e3_rows) and baseline_spread is not None and baseline_spread <= 0.01

    e1_supported = bool(e1) and e1["address_reduction_pct"] > e1["token_channel_reduction_pct"]
    accuracy_supported = bool(address) and address["mean_delta_top1"] > 0 and all(
        address["mean_delta_top1"] > row["mean_delta_top1"] for row in controls
    )
    address_replacements = [
        float(row["replacement_top1"])
        for row in e3_rows
        if row.get("mode") == "address_lut" and row.get("replacement_top1") is not None
    ]
    beats_training_best = bool(address_replacements) and baseline_best is not None and max(address_replacements) > float(baseline_best)
    decision = (
        "PASS"
        if e1_supported and protocol_valid and accuracy_supported and beats_training_best
        else "PARTIAL"
        if e1_supported
        else "FAIL"
    )
    interpretation = {
        "PASS": (
            "T=4 CIFAR-100 supports both address-specific reconstruction and a positive "
            "address-LUT adapter accuracy signal over adjacent controls."
        ),
        "PARTIAL": (
            "T=4 CIFAR-100 supports interpretable address reconstruction, but the adapter "
            "does not yet prove a stronger accuracy claim over controls."
        ),
        "FAIL": (
            "T=4 CIFAR-100 does not yet provide the required interpretable address evidence."
        ),
    }[decision]

    report = {
        "decision": decision,
        "train_dir": str(train_dir) if train_dir else None,
        "train_manifest": train_manifest,
        "baseline_best_top1": baseline_best,
        "train_summary": train_summary,
        "e1": e1,
        "e3_rows": e3_rows,
        "e3_summary": e3_summary,
        "checks": {
            "has_t4_train": train_dir is not None,
            "e1_address_beats_token": e1_supported,
            "address_adapter_beats_controls": accuracy_supported,
            "protocol_valid": protocol_valid,
            "baseline_spread": baseline_spread,
            "address_beats_training_best": beats_training_best,
        },
        "interpretation": interpretation,
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    (args.output_prefix.with_suffix(".json")).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_prefix.with_name(args.output_prefix.name + "_e3_rows.csv"), e3_rows)
    _write_csv(args.output_prefix.with_name(args.output_prefix.name + "_e3_summary.csv"), e3_summary)
    (args.output_prefix.with_suffix(".md")).write_text(_markdown(report), encoding="utf-8")
    print(f"decision={decision}")
    print(f"json={args.output_prefix.with_suffix('.json')}")
    print(f"markdown={args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
