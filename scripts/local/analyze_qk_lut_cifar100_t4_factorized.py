#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MODES = {
    "address_lut",
    "global_mean",
    "token_channel_lut",
    "shuffled_address_lut",
    "factorized_sum_lut",
    "factorized_gated_lut",
    "matched_param_address_lut",
    "factorized_shuffled_lut",
}
CANDIDATES = ("factorized_sum_lut", "factorized_gated_lut")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def training_best(results_root: Path) -> float | None:
    values = []
    pattern = re.compile(r"Best metric:\s*([0-9.]+)")
    for result_dir in results_root.glob("qkformer_cifar100_train_*"):
        manifest = load_json(result_dir / "checkpoint_manifest.json")
        if str(manifest.get("time_step")) != "4":
            continue
        log_path = result_dir / "train_log.txt"
        if log_path.exists():
            values.extend(float(value) for value in pattern.findall(log_path.read_text(errors="replace")))
    return max(values) if values else None


def load_runs(results_root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in results_root.glob("qkformer_lut_e3_trainable_lut_*/metrics.json"):
        metrics = load_json(path)
        model = metrics.get("model", {})
        protocol = metrics.get("protocol", {})
        adapter = metrics.get("adapter_summary", {})
        adapter_config = metrics.get("adapter_config", {})
        train = metrics.get("train_config", {})
        mode = adapter.get("mode")
        if model.get("family") != "cifar100" or str(model.get("time_step")) != "4":
            continue
        if mode not in MODES or int(protocol.get("version", 0)) < 4:
            continue
        if not protocol.get("evaluation_amp") or protocol.get("evaluation_loader") != "timm":
            continue
        if abs(float(adapter_config.get("alpha_init", -1)) - 0.025) > 1e-9 or int(train.get("epochs", -1)) != 2:
            continue
        classification = metrics.get("classification", {})
        baseline = classification.get("baseline", {})
        replacement = classification.get("replacement", {})
        delta = classification.get("delta", {})
        runs.append(
            {
                "result_dir": str(path.parent),
                "mode": mode,
                "seed": int(metrics.get("experiment", {}).get("seed", -1)),
                "parameters": int(adapter.get("trainable_parameters", 0)),
                "baseline_top1": float(baseline.get("top1")),
                "replacement_top1": float(replacement.get("top1")),
                "delta_top1": float(delta.get("top1")),
                "delta_loss": float(delta.get("loss")),
                "local_mse": float(replacement.get("local_mse")),
                "logit_mse": float(replacement.get("logit_mse")),
                "kl": float(replacement.get("kl_to_baseline")),
            }
        )
    return sorted(runs, key=lambda row: (row["mode"], row["seed"], row["result_dir"]))


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run["mode"]].append(run)
    rows = []
    for mode, items in sorted(groups.items()):
        latest_by_seed = {}
        for item in items:
            latest_by_seed[item["seed"]] = item
        selected = [latest_by_seed[seed] for seed in sorted(latest_by_seed)]
        if not selected:
            continue
        rows.append(
            {
                "mode": mode,
                "n": len(selected),
                "mean_delta_top1": statistics.mean(item["delta_top1"] for item in selected),
                "min_delta_top1": min(item["delta_top1"] for item in selected),
                "max_delta_top1": max(item["delta_top1"] for item in selected),
                "mean_replacement_top1": statistics.mean(item["replacement_top1"] for item in selected),
                "max_replacement_top1": max(item["replacement_top1"] for item in selected),
                "mean_delta_loss": statistics.mean(item["delta_loss"] for item in selected),
                "mean_local_mse": statistics.mean(item["local_mse"] for item in selected),
                "mean_logit_mse": statistics.mean(item["logit_mse"] for item in selected),
                "mean_kl": statistics.mean(item["kl"] for item in selected),
                "parameters": selected[0]["parameters"],
            }
        )
    return rows


def main() -> None:
    results_root = Path("results")
    runs = load_runs(results_root)
    summary = summarize(runs)
    by_mode = {row["mode"]: row for row in summary}
    best = training_best(results_root)
    candidate_reports = []
    for candidate in CANDIDATES:
        row = by_mode.get(candidate)
        controls = [by_mode.get(mode) for mode in ("address_lut", "global_mean", "matched_param_address_lut", "factorized_shuffled_lut")]
        complete = row is not None and row["n"] >= 3 and all(control is not None and control["n"] >= 3 for control in controls)
        checks = {
            "three_seeds": bool(row and row["n"] >= 3),
            "positive_mean_delta": bool(row and row["mean_delta_top1"] > 0),
            "beats_all_controls": bool(row and controls and all(row["mean_delta_top1"] > control["mean_delta_top1"] for control in controls if control)),
            "beats_training_best": bool(row and best is not None and row["max_replacement_top1"] > best),
            "beats_shuffled_local_mse": bool(row and by_mode.get("factorized_shuffled_lut") and row["mean_local_mse"] < by_mode["factorized_shuffled_lut"]["mean_local_mse"]),
        }
        candidate_reports.append({"candidate": candidate, "complete": complete, "checks": checks, "pass": complete and all(checks.values())})
    if any(item["pass"] for item in candidate_reports):
        decision = "PASS"
    elif any(item["complete"] for item in candidate_reports):
        decision = "FAIL"
    else:
        decision = "PENDING"

    output = {
        "decision": decision,
        "training_best_top1": best,
        "summary": summary,
        "candidates": candidate_reports,
        "num_protocol_runs": len(runs),
    }
    prefix = results_root / "qk_lutformer_cifar100_t4_factorized"
    prefix.with_suffix(".json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()) if summary else ["mode"])
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# CIFAR-100 T=4 Factorized LUT Pilot", "", f"**Decision:** `{decision}`", "", "| Mode | n | Mean delta Acc@1 | Min | Max | Mean Acc@1 | Params |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['mode']} | {row['n']} | {row['mean_delta_top1']:.4f} | {row['min_delta_top1']:.4f} | {row['max_delta_top1']:.4f} | {row['mean_replacement_top1']:.4f} | {row['parameters']} |")
    lines.extend(["", "A PASS requires a three-seed factorized candidate to improve mean accuracy, beat all adjacent controls, exceed the saved training best in at least one seed, and beat the shuffled factorized control in local MSE.", ""])
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"decision={decision}")
    print(f"json={prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
