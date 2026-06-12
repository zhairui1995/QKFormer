#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _best_train_dir(results_root: Path) -> Optional[Path]:
    candidates = []
    for path in results_root.glob("qkformer_cifar100_train_*"):
        manifest = _load_json(path / "checkpoint_manifest.json")
        if str(manifest.get("time_step")) == "4":
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


def _checkpoint_path(metrics: Dict[str, Any]) -> str:
    return str(metrics.get("model", {}).get("checkpoint", {}).get("path") or "")


def _matches_c100_t4(metrics: Dict[str, Any], train_dir: Optional[Path]) -> bool:
    model = metrics.get("model", {})
    if model.get("family") != "cifar100" or str(model.get("time_step")) != "4":
        return False
    if train_dir is None:
        return True
    return train_dir.name in Path(_checkpoint_path(metrics)).parts


def _per_sample_path(metrics_path: Path, metrics: Dict[str, Any]) -> Optional[Path]:
    recorded = str(metrics.get("classification", {}).get("per_sample", {}).get("path") or "")
    candidates = []
    if recorded:
        candidates.append(Path(recorded))
        candidates.append(metrics_path.parent / Path(recorded).name)
    candidates.append(metrics_path.parent / "per_sample_predictions.csv")
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in csv.DictReader(path.open("r", encoding="utf-8")):
        rows.append(
            {
                "sample_index": int(row["sample_index"]),
                "target": int(row["target"]),
                "baseline_correct": int(row["baseline_correct"]),
                "replacement_correct": int(row["replacement_correct"]),
                "baseline_ce": float(row["baseline_ce"]),
                "replacement_ce": float(row["replacement_ce"]),
                "baseline_margin": float(row["baseline_margin"]),
                "replacement_margin": float(row["replacement_margin"]),
            }
        )
    return rows


def _mean(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def _margin_bin(value: float) -> str:
    if value <= 0.0:
        return "wrong_or_negative"
    if value <= 0.5:
        return "low_0_0p5"
    if value <= 1.5:
        return "mid_0p5_1p5"
    if value <= 3.0:
        return "high_1p5_3"
    return "very_high_gt3"


def _summarize_rows(rows: List[Dict[str, Any]], margin_eps: float) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    baseline_correct = [row["baseline_correct"] for row in rows]
    replacement_correct = [row["replacement_correct"] for row in rows]
    baseline_ce = [row["baseline_ce"] for row in rows]
    replacement_ce = [row["replacement_ce"] for row in rows]
    ce_select = [row["replacement_ce"] < row["baseline_ce"] for row in rows]
    margin_select = [
        row["replacement_margin"] > row["baseline_margin"] + margin_eps for row in rows
    ]

    def selected_acc(selector: List[bool]) -> float:
        chosen = [
            row["replacement_correct"] if use_replacement else row["baseline_correct"]
            for row, use_replacement in zip(rows, selector)
        ]
        return 100.0 * sum(chosen) / len(chosen)

    def selected_loss(selector: List[bool]) -> float:
        chosen = [
            row["replacement_ce"] if use_replacement else row["baseline_ce"]
            for row, use_replacement in zip(rows, selector)
        ]
        return sum(chosen) / len(chosen)

    bins: Dict[str, List[Tuple[Dict[str, Any], bool]]] = defaultdict(list)
    for row, use_replacement in zip(rows, ce_select):
        bins[_margin_bin(row["baseline_margin"])].append((row, use_replacement))
    bin_rows = []
    for name in ["wrong_or_negative", "low_0_0p5", "mid_0p5_1p5", "high_1p5_3", "very_high_gt3"]:
        items = bins.get(name, [])
        if not items:
            continue
        item_rows = [item[0] for item in items]
        item_select = [item[1] for item in items]
        base_acc = 100.0 * sum(row["baseline_correct"] for row in item_rows) / len(item_rows)
        oracle_acc = 100.0 * sum(
            row["replacement_correct"] if use else row["baseline_correct"]
            for row, use in zip(item_rows, item_select)
        ) / len(item_rows)
        bin_rows.append(
            {
                "margin_bin": name,
                "n": len(item_rows),
                "coverage_pct": 100.0 * len(item_rows) / n,
                "baseline_acc": base_acc,
                "replacement_acc": 100.0
                * sum(row["replacement_correct"] for row in item_rows)
                / len(item_rows),
                "oracle_ce_acc": oracle_acc,
                "oracle_ce_gain": oracle_acc - base_acc,
                "ce_selected_pct": 100.0 * sum(item_select) / len(item_select),
            }
        )

    baseline_acc = 100.0 * sum(baseline_correct) / n
    replacement_acc = 100.0 * sum(replacement_correct) / n
    oracle_ce_acc = selected_acc(ce_select)
    oracle_margin_acc = selected_acc(margin_select)
    return {
        "n": n,
        "baseline_acc": baseline_acc,
        "replacement_acc": replacement_acc,
        "replacement_delta_acc": replacement_acc - baseline_acc,
        "baseline_loss": sum(baseline_ce) / n,
        "replacement_loss": sum(replacement_ce) / n,
        "replacement_delta_loss": sum(replacement_ce) / n - sum(baseline_ce) / n,
        "oracle_ce_acc": oracle_ce_acc,
        "oracle_ce_gain": oracle_ce_acc - baseline_acc,
        "oracle_ce_loss": selected_loss(ce_select),
        "oracle_ce_loss_delta": selected_loss(ce_select) - sum(baseline_ce) / n,
        "oracle_ce_selected_pct": 100.0 * sum(ce_select) / n,
        "oracle_margin_acc": oracle_margin_acc,
        "oracle_margin_gain": oracle_margin_acc - baseline_acc,
        "oracle_margin_loss": selected_loss(margin_select),
        "oracle_margin_selected_pct": 100.0 * sum(margin_select) / n,
        "harmful_rate_pct": 100.0
        * sum(1 for row in rows if row["baseline_correct"] and not row["replacement_correct"])
        / n,
        "rescue_rate_pct": 100.0
        * sum(1 for row in rows if not row["baseline_correct"] and row["replacement_correct"])
        / n,
        "ce_selected_harmful_rate_pct": 100.0
        * sum(
            1
            for row, use in zip(rows, ce_select)
            if use and row["baseline_correct"] and not row["replacement_correct"]
        )
        / n,
        "ce_selected_rescue_rate_pct": 100.0
        * sum(
            1
            for row, use in zip(rows, ce_select)
            if use and not row["baseline_correct"] and row["replacement_correct"]
        )
        / n,
        "margin_bins": bin_rows,
    }


def _collect_runs(
    results_root: Path,
    train_dir: Optional[Path],
    margin_eps: float,
    alpha: Optional[float],
    epochs: Optional[int],
    calibration_batches: Optional[int],
    train_batches: Optional[int],
) -> List[Dict[str, Any]]:
    runs = []
    for metrics_path in results_root.glob("qkformer_lut_e3_trainable_lut_*/metrics.json"):
        metrics = _load_json(metrics_path)
        if not _matches_c100_t4(metrics, train_dir):
            continue
        path = _per_sample_path(metrics_path, metrics)
        if path is None:
            continue
        protocol = metrics.get("protocol", {})
        if int(protocol.get("version", 0)) < 4:
            continue
        adapter = metrics.get("adapter_summary", {})
        adapter_config = metrics.get("adapter_config", {})
        train_config = metrics.get("train_config", {})
        data = metrics.get("data", {})
        mode = str(adapter.get("mode") or metrics.get("adapter_config", {}).get("mode") or "")
        if not mode:
            continue
        run_alpha = adapter_config.get("alpha_init")
        run_epochs = train_config.get("epochs")
        run_calib_batches = data.get("calibration", {}).get("num_batches")
        run_train_batches = data.get("train", {}).get("num_batches")
        if alpha is not None and (run_alpha is None or abs(float(run_alpha) - float(alpha)) > 1e-12):
            continue
        if epochs is not None and int(run_epochs) != int(epochs):
            continue
        if calibration_batches is not None and int(run_calib_batches) != int(calibration_batches):
            continue
        if train_batches is not None and int(run_train_batches) != int(train_batches):
            continue
        rows = _read_rows(path)
        summary = _summarize_rows(rows, margin_eps)
        runs.append(
            {
                "result_dir": str(metrics_path.parent),
                "per_sample_path": str(path),
                "mode": mode,
                "seed": metrics.get("experiment", {}).get("seed"),
                "target_modules": ",".join(adapter.get("target_modules", []) or []),
                "alpha": run_alpha,
                "epochs": run_epochs,
                "calibration_batches": run_calib_batches,
                "train_batches": run_train_batches,
                "eval_loader": protocol.get("evaluation_loader"),
                "eval_amp": protocol.get("evaluation_amp"),
                **summary,
            }
        )
    return sorted(runs, key=lambda row: (row["mode"], str(row["seed"]), row["result_dir"]))


def _summarize_modes(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[row["mode"]].append(row)
    summary = []
    keys = [
        "baseline_acc",
        "replacement_acc",
        "replacement_delta_acc",
        "oracle_ce_acc",
        "oracle_ce_gain",
        "oracle_ce_loss_delta",
        "oracle_ce_selected_pct",
        "oracle_margin_acc",
        "oracle_margin_gain",
        "harmful_rate_pct",
        "rescue_rate_pct",
        "ce_selected_harmful_rate_pct",
        "ce_selected_rescue_rate_pct",
    ]
    for mode, items in sorted(grouped.items()):
        row = {"mode": mode, "n": len(items), "seeds": ",".join(str(item["seed"]) for item in items)}
        for key in keys:
            value = _mean(float(item[key]) for item in items if item.get(key) is not None)
            row[f"mean_{key}"] = value
            values = [float(item[key]) for item in items if item.get(key) is not None]
            if values and key in {"oracle_ce_gain", "replacement_delta_acc"}:
                row[f"min_{key}"] = min(values)
                row[f"max_{key}"] = max(values)
        summary.append(row)
    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in keys}
            if isinstance(flat.get("margin_bins"), list):
                flat["margin_bins"] = json.dumps(flat["margin_bins"], sort_keys=True)
            writer.writerow(flat)


def _decision(summary: List[Dict[str, Any]]) -> Tuple[str, str]:
    by_mode = {row["mode"]: row for row in summary}
    aligned = by_mode.get("global_plus_address_lut") or by_mode.get("address_lut")
    shuffled = by_mode.get("global_plus_shuffled_address_lut") or by_mode.get("shuffled_address_lut")
    if not aligned:
        return "MISSING", "No aligned address-LUT per-sample run was found."
    aligned_gain = float(aligned.get("mean_oracle_ce_gain") or 0.0)
    shuffled_gain = float(shuffled.get("mean_oracle_ce_gain") or 0.0) if shuffled else None
    gap = aligned_gain - shuffled_gain if shuffled_gain is not None else None
    if aligned_gain >= 0.30 and (gap is None or gap >= 0.10):
        return (
            "ORACLE-GO",
            "The aligned LUT has enough oracle selective headroom to justify a deterministic gate experiment.",
        )
    if aligned_gain >= 0.05 and (gap is None or gap >= 0.0):
        return (
            "CONDITIONAL-ORACLE",
            "The aligned LUT has some oracle headroom, but the ceiling is modest or not clearly above shuffled.",
        )
    return (
        "NO-GO",
        "The oracle upper bound is too small to justify more accuracy-oriented adapter training.",
    )


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# CIFAR-100 T=4 Selective Oracle Gate",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        report["interpretation"],
        "",
        "This is an upper-bound diagnostic, not a deployable method result.",
        "",
        "## Mode Summary",
        "| Mode | n | Replacement ΔAcc@1 | Oracle CE gain | Oracle selected | Harmful | Rescue |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("mode_summary", []):
        lines.append(
            "| {mode} | {n} | {rd} | {og} | {sel} | {harm} | {rescue} |".format(
                mode=row["mode"],
                n=row["n"],
                rd="-" if row.get("mean_replacement_delta_acc") is None else f"{row['mean_replacement_delta_acc']:.4f}",
                og="-" if row.get("mean_oracle_ce_gain") is None else f"{row['mean_oracle_ce_gain']:.4f}",
                sel="-" if row.get("mean_oracle_ce_selected_pct") is None else f"{row['mean_oracle_ce_selected_pct']:.2f}%",
                harm="-" if row.get("mean_harmful_rate_pct") is None else f"{row['mean_harmful_rate_pct']:.2f}%",
                rescue="-" if row.get("mean_rescue_rate_pct") is None else f"{row['mean_rescue_rate_pct']:.2f}%",
            )
        )
    lines.extend(["", "## Claim Boundary", ""])
    lines.extend(
        [
            "- `ORACLE-GO`: continue to a calibration-only deterministic gate.",
            "- `CONDITIONAL-ORACLE`: run only a small deterministic gate probe; do not claim accuracy improvement.",
            "- `NO-GO`: stop accuracy tuning and keep the paper on audit/reconstruction.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-sample selective oracle gate diagnostics")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/qk_lutformer_cifar100_t4_selective_oracle"),
    )
    parser.add_argument("--margin-eps", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--calibration-batches", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=128)
    args = parser.parse_args()

    train_dir = _best_train_dir(args.results_root)
    runs = _collect_runs(
        args.results_root,
        train_dir,
        args.margin_eps,
        args.alpha,
        args.epochs,
        args.calibration_batches,
        args.train_batches,
    )
    summary = _summarize_modes(runs)
    decision, interpretation = _decision(summary)
    report = {
        "decision": decision,
        "interpretation": interpretation,
        "train_dir": str(train_dir) if train_dir else None,
        "margin_eps": args.margin_eps,
        "filters": {
            "alpha": args.alpha,
            "epochs": args.epochs,
            "calibration_batches": args.calibration_batches,
            "train_batches": args.train_batches,
        },
        "runs": runs,
        "mode_summary": summary,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    (args.output_prefix.with_suffix(".json")).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_prefix.with_name(args.output_prefix.name + "_runs.csv"), runs)
    _write_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    (args.output_prefix.with_suffix(".md")).write_text(_markdown(report), encoding="utf-8")
    print(f"decision={decision}")
    print(f"json={args.output_prefix.with_suffix('.json')}")
    print(f"markdown={args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
