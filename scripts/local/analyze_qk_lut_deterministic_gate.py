#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MODES = (
    "global_plus_address_lut",
    "global_plus_shuffled_address_lut",
    "global_mean",
)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _prediction_path(result_dir: Path, section: Dict[str, Any], fallback: str) -> Optional[Path]:
    recorded = str(section.get("per_sample", {}).get("path") or "")
    candidates = [result_dir / fallback]
    if recorded:
        candidates.insert(0, result_dir / Path(recorded).name)
        candidates.insert(0, Path(recorded))
    return next((path for path in candidates if path.exists()), None)


def _read_rows(path: Path) -> List[Dict[str, float]]:
    rows = []
    for row in csv.DictReader(path.open("r", encoding="utf-8")):
        rows.append(
            {
                "baseline_correct": int(row["baseline_correct"]),
                "replacement_correct": int(row["replacement_correct"]),
                "baseline_ce": float(row["baseline_ce"]),
                "replacement_ce": float(row["replacement_ce"]),
                "baseline_pred_margin": float(row["baseline_pred_margin"]),
                "replacement_pred_margin": float(row["replacement_pred_margin"]),
            }
        )
    return rows


def _score(row: Dict[str, float]) -> float:
    return row["replacement_pred_margin"] - row["baseline_pred_margin"]


def _quantile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.inf
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _evaluate_gate(rows: List[Dict[str, float]], threshold: float) -> Dict[str, float]:
    selectors = [_score(row) >= threshold for row in rows]
    n = len(rows)
    baseline_acc = 100.0 * sum(row["baseline_correct"] for row in rows) / n
    replacement_acc = 100.0 * sum(row["replacement_correct"] for row in rows) / n
    gated_acc = 100.0 * sum(
        row["replacement_correct"] if use else row["baseline_correct"]
        for row, use in zip(rows, selectors)
    ) / n
    baseline_loss = sum(row["baseline_ce"] for row in rows) / n
    gated_loss = sum(
        row["replacement_ce"] if use else row["baseline_ce"]
        for row, use in zip(rows, selectors)
    ) / n
    selected = sum(selectors)
    return {
        "baseline_acc": baseline_acc,
        "replacement_acc": replacement_acc,
        "gated_acc": gated_acc,
        "gated_delta_acc": gated_acc - baseline_acc,
        "baseline_loss": baseline_loss,
        "gated_loss": gated_loss,
        "gated_delta_loss": gated_loss - baseline_loss,
        "selected_pct": 100.0 * selected / n,
        "selected_n": selected,
        "n": n,
    }


def _fit_threshold(rows: List[Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    scores = [_score(row) for row in rows]
    quantiles = [i / 20.0 for i in range(21)]
    span = max(scores) - min(scores)
    epsilon = max(1e-12, span * 1e-9)
    candidates = {
        max(scores) + epsilon,
        min(scores) - epsilon,
        *(_quantile(scores, q) for q in quantiles),
    }
    evaluated = [(threshold, _evaluate_gate(rows, threshold)) for threshold in candidates]
    # Accuracy is the registered objective. Ties prefer lower replacement coverage,
    # then the stricter threshold, so the gate cannot win by gratuitous replacement.
    return max(
        evaluated,
        key=lambda item: (
            item[1]["gated_acc"],
            -item[1]["selected_pct"],
            item[0],
        ),
    )


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def _collect_runs(results_root: Path, alpha: float, epochs: int, batches: int) -> List[Dict[str, Any]]:
    runs = []
    for metrics_path in results_root.glob("qkformer_lut_e3_trainable_lut_*/metrics.json"):
        metrics = _load_json(metrics_path)
        model = metrics.get("model", {})
        protocol = metrics.get("protocol", {})
        if model.get("family") != "cifar100" or str(model.get("time_step")) != "4":
            continue
        if int(protocol.get("version", 0)) < 5:
            continue
        mode = str(metrics.get("adapter_summary", {}).get("mode") or "")
        if mode not in MODES:
            continue
        adapter_cfg = metrics.get("adapter_config", {})
        train_cfg = metrics.get("train_config", {})
        data_cfg = metrics.get("data", {})
        if abs(float(adapter_cfg.get("alpha_init", -1.0)) - alpha) > 1e-12:
            continue
        if int(train_cfg.get("epochs", -1)) != epochs:
            continue
        if int(data_cfg.get("calibration", {}).get("num_batches", -1)) != batches:
            continue
        if int(data_cfg.get("train", {}).get("num_batches", -1)) != batches:
            continue
        if int(data_cfg.get("gate_calibration", {}).get("num_batches", -1)) != batches:
            continue

        result_dir = metrics_path.parent
        gate_section = metrics.get("gate_calibration") or {}
        eval_section = metrics.get("classification") or {}
        gate_path = _prediction_path(result_dir, gate_section, "gate_calibration_predictions.csv")
        eval_path = _prediction_path(result_dir, eval_section, "per_sample_predictions.csv")
        if gate_path is None or eval_path is None:
            continue
        gate_rows = _read_rows(gate_path)
        eval_rows = _read_rows(eval_path)
        threshold, gate_metrics = _fit_threshold(gate_rows)
        eval_metrics = _evaluate_gate(eval_rows, threshold)
        runs.append(
            {
                "result_dir": str(result_dir),
                "mtime": result_dir.stat().st_mtime,
                "mode": mode,
                "seed": int(metrics.get("experiment", {}).get("seed", -1)),
                "threshold": threshold,
                "gate_calibration_acc": gate_metrics["gated_acc"],
                "gate_calibration_delta_acc": gate_metrics["gated_delta_acc"],
                "gate_calibration_selected_pct": gate_metrics["selected_pct"],
                **{f"validation_{key}": value for key, value in eval_metrics.items()},
            }
        )
    latest: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in runs:
        key = (row["mode"], row["seed"])
        if key not in latest or row["mtime"] > latest[key]["mtime"]:
            latest[key] = row
    return sorted(latest.values(), key=lambda row: (row["mode"], row["seed"]))


def _summarize(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[row["mode"]].append(row)
    result = []
    for mode, rows in sorted(grouped.items()):
        deltas = [float(row["validation_gated_delta_acc"]) for row in rows]
        result.append(
            {
                "mode": mode,
                "n": len(rows),
                "seeds": ",".join(str(row["seed"]) for row in rows),
                "mean_baseline_acc": _mean(float(row["validation_baseline_acc"]) for row in rows),
                "mean_replacement_acc": _mean(float(row["validation_replacement_acc"]) for row in rows),
                "mean_gated_acc": _mean(float(row["validation_gated_acc"]) for row in rows),
                "mean_gated_delta_acc": _mean(deltas),
                "min_gated_delta_acc": min(deltas),
                "max_gated_delta_acc": max(deltas),
                "mean_gated_delta_loss": _mean(float(row["validation_gated_delta_loss"]) for row in rows),
                "mean_selected_pct": _mean(float(row["validation_selected_pct"]) for row in rows),
                "mean_gate_calibration_delta_acc": _mean(
                    float(row["gate_calibration_delta_acc"]) for row in rows
                ),
            }
        )
    return result


def _decision(summary: List[Dict[str, Any]], benchmark: float) -> Tuple[str, str]:
    by_mode = {row["mode"]: row for row in summary}
    aligned = by_mode.get("global_plus_address_lut")
    shuffled = by_mode.get("global_plus_shuffled_address_lut")
    global_control = by_mode.get("global_mean")
    if not aligned or aligned["n"] < 3 or not shuffled or not global_control:
        return "MISSING", "The registered three-seed aligned/shuffled/global comparison is incomplete."
    aligned_acc = float(aligned["mean_gated_acc"])
    aligned_delta = float(aligned["mean_gated_delta_acc"])
    control_acc = max(float(shuffled["mean_gated_acc"]), float(global_control["mean_gated_acc"]))
    gap = aligned_acc - control_acc
    if aligned_acc > benchmark and aligned_delta > 0.0 and float(aligned["min_gated_delta_acc"]) >= 0.0 and gap >= 0.05:
        return "PASS", "The calibration-only gate is stable, exceeds the saved QKFormer best, and separates from matched controls."
    if aligned_delta > 0.0 and gap > 0.0:
        return "PARTIAL", "The gate is directionally positive but misses the preregistered stability, benchmark, or control-separation requirement."
    return "FAIL", "The deterministic gate does not convert oracle headroom into an address-specific validation gain. Stop accuracy tuning."


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# CIFAR-100 T=4 Deterministic Risk Gate",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        report["interpretation"],
        "",
        "Thresholds are fit only on a disjoint CIFAR-100 train partition. Validation labels are never used for gate selection.",
        "",
        "| Mode | n | Baseline | Replacement | Gated | Gated delta | Min delta | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["mode_summary"]:
        lines.append(
            f"| {row['mode']} | {row['n']} | {row['mean_baseline_acc']:.4f} | "
            f"{row['mean_replacement_acc']:.4f} | {row['mean_gated_acc']:.4f} | "
            f"{row['mean_gated_delta_acc']:+.4f} | {row['min_gated_delta_acc']:+.4f} | "
            f"{row['mean_selected_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Registered Gate",
            "",
            "The score is replacement top-1/top-2 margin minus baseline top-1/top-2 margin. A single threshold is selected from 5% calibration quantiles to maximize calibration accuracy; ties prefer lower replacement coverage.",
            "",
            "## Claim Boundary",
            "",
            "- `PASS`: eligible for a cautious accuracy result.",
            "- `PARTIAL`: report as exploratory evidence only.",
            "- `FAIL`: keep accuracy as a limits result and end this tuning branch.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a calibration-only deterministic QK-LUT gate")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/qk_lutformer_cifar100_t4_deterministic_gate"),
    )
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--benchmark", type=float, default=81.23)
    args = parser.parse_args()

    runs = _collect_runs(args.results_root, args.alpha, args.epochs, args.batches)
    summary = _summarize(runs)
    decision, interpretation = _decision(summary, args.benchmark)
    report = {
        "decision": decision,
        "interpretation": interpretation,
        "benchmark_acc": args.benchmark,
        "registered_protocol": {
            "score": "replacement_pred_margin_minus_baseline_pred_margin",
            "threshold_grid": "calibration_quantiles_step_0.05_plus_all_and_none",
            "selection_objective": "gate_calibration_accuracy",
            "tie_break": "lower_replacement_coverage_then_stricter_threshold",
            "partition": "disjoint_train_indices_for_prototype_adapter_and_gate",
        },
        "runs": runs,
        "mode_summary": summary,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    _write_csv(args.output_prefix.with_name(args.output_prefix.name + "_runs.csv"), runs)
    _write_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    print(f"decision={decision}")
    print(f"json={args.output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
