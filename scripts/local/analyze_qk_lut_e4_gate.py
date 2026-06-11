#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CANDIDATES = ("factorized_sum_lut", "factorized_gated_lut")
REQUIRED_MODES = {
    "address_lut",
    "global_mean",
    "matched_param_address_lut",
    "factorized_shuffled_lut",
}


def load_runs(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    runs = []
    for path in paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        mode = metrics.get("adapter_config", {}).get("mode")
        if mode not in REQUIRED_MODES | set(CANDIDATES):
            continue
        family = metrics.get("model", {}).get("family")
        checkpoint = metrics.get("model", {}).get("checkpoint", {}).get("path") or "unknown"
        seed = int(metrics.get("data", {}).get("calibration", {}).get("seed", -1))
        evaluation = metrics["classification"]
        checkpoint_path = Path(checkpoint)
        checkpoint_id = "/".join(checkpoint_path.parts[-2:]) if checkpoint != "unknown" else checkpoint
        runs.append(
            {
                "path": str(path),
                "family": family,
                "checkpoint": checkpoint_id,
                "seed": seed,
                "mode": mode,
                "local_mse": float(evaluation["replacement"]["local_mse"]),
                "logit_mse": float(evaluation["replacement"]["logit_mse"]),
                "kl": float(evaluation["replacement"]["kl_to_baseline"]),
                "loss_drift": abs(float(evaluation["delta"]["loss"])),
                "top1_delta": float(evaluation["delta"]["top1"]),
                "parameters": int(metrics["adapter_summary"]["trainable_parameters"]),
            }
        )
    return runs


def paired_standard_error(values: List[float]) -> float:
    if len(values) < 2:
        return math.inf
    return statistics.stdev(values) / math.sqrt(len(values))


def evaluate_candidate(candidate: str, groups: Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    pairs = []
    for key, modes in sorted(groups.items()):
        needed = REQUIRED_MODES | {candidate}
        if not needed.issubset(modes):
            continue
        factorized = modes[candidate]
        address = modes["address_lut"]
        global_control = modes["global_mean"]
        matched = modes["matched_param_address_lut"]
        shuffled = modes["factorized_shuffled_lut"]
        metric_wins = {
            metric: factorized[metric] < address[metric]
            for metric in ("local_mse", "logit_mse", "kl", "loss_drift")
        }
        pairs.append(
            {
                "key": {"family": key[0], "checkpoint": key[1], "seed": key[2]},
                "metric_wins": metric_wins,
                "paired_win": all(metric_wins.values()),
                "local_mse_improvement": address["local_mse"] - factorized["local_mse"],
                "parameter_budget_ok": factorized["parameters"] <= matched["parameters"],
                "accuracy_budget_ok": factorized["top1_delta"] >= global_control["top1_delta"] - 0.05,
                "shuffle_control_ok": factorized["local_mse"] < shuffled["local_mse"],
                "factorized": factorized,
                "address": address,
            }
        )

    c10_pairs = [pair for pair in pairs if pair["key"]["family"] == "cifar10"]
    c100_pairs = [pair for pair in pairs if pair["key"]["family"] == "cifar100"]
    improvements = [pair["local_mse_improvement"] for pair in c10_pairs]
    mean_improvement = statistics.mean(improvements) if improvements else 0.0
    standard_error = paired_standard_error(improvements)
    complete = len(c10_pairs) >= 6
    checks = {
        "at_least_5_of_6_paired_wins": sum(pair["paired_win"] for pair in c10_pairs) >= 5,
        "mean_improvement_gt_2se": mean_improvement > 2.0 * standard_error,
        "parameter_budget": bool(c10_pairs) and all(pair["parameter_budget_ok"] for pair in c10_pairs),
        "accuracy_budget": bool(c10_pairs) and all(pair["accuracy_budget_ok"] for pair in c10_pairs),
        "shuffle_control": sum(pair["shuffle_control_ok"] for pair in c10_pairs) >= 5,
    }
    pilot_improvements = [pair["local_mse_improvement"] for pair in c100_pairs]
    pilot_mean = statistics.mean(pilot_improvements) if pilot_improvements else 0.0
    pilot_se = paired_standard_error(pilot_improvements)
    pilot_checks = {
        "three_paired_wins": len(c100_pairs) >= 3 and all(pair["paired_win"] for pair in c100_pairs),
        "mean_improvement_gt_2se": pilot_mean > 2.0 * pilot_se,
        "parameter_budget": len(c100_pairs) >= 3 and all(pair["parameter_budget_ok"] for pair in c100_pairs),
        "accuracy_budget": len(c100_pairs) >= 3 and all(pair["accuracy_budget_ok"] for pair in c100_pairs),
        "shuffle_control": len(c100_pairs) >= 3 and all(pair["shuffle_control_ok"] for pair in c100_pairs),
    }
    return {
        "candidate": candidate,
        "complete": complete,
        "num_cifar10_pairs": len(c10_pairs),
        "num_cifar100_pilot_pairs": len(c100_pairs),
        "paired_wins": sum(pair["paired_win"] for pair in c10_pairs),
        "mean_local_mse_improvement": mean_improvement,
        "paired_standard_error": standard_error,
        "checks": checks,
        "pass": complete and all(checks.values()),
        "pilot_mean_local_mse_improvement": pilot_mean,
        "pilot_paired_standard_error": pilot_se,
        "pilot_checks": pilot_checks,
        "pilot_pass": len(c100_pairs) >= 3 and all(pilot_checks.values()),
        "pairs": pairs,
    }


def markdown(report: Dict[str, Any]) -> str:
    lines = ["# QK-LUTFormer E4 Gate Report", "", f"**Decision:** `{report['decision']}`", ""]
    for candidate in report["candidates"]:
        lines.extend(
            [
                f"## {candidate['candidate']}",
                "",
                f"- Complete CIFAR-10 pairs: {candidate['num_cifar10_pairs']}/6",
                f"- Complete CIFAR-100 pilot pairs: {candidate['num_cifar100_pilot_pairs']}/3",
                f"- CIFAR-100 pilot pass: {candidate['pilot_pass']}",
                f"- Paired wins: {candidate['paired_wins']}/6",
                f"- Mean local-MSE improvement: {candidate['mean_local_mse_improvement']:.8g}",
                f"- Paired standard error: {candidate['paired_standard_error']:.8g}",
                f"- Gate pass: {candidate['pass']}",
                "",
                "| Check | Pass |",
                "|---|---|",
            ]
        )
        for name, passed in candidate["checks"].items():
            lines.append(f"| {name} | {passed} |")
        lines.extend(["", "### CIFAR-100 pilot checks", "", "| Check | Pass |", "|---|---|"])
        for name, passed in candidate["pilot_checks"].items():
            lines.append(f"| {name} | {passed} |")
        lines.append("")
    lines.extend(
        [
            "The final gate requires six CIFAR-10 pairs from checkpoint seeds 43/44 crossed with three adapter seeds. "
            "CIFAR-100 rows are treated as the stage-1 pilot and do not replace the six-run replication gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the QK-LUTFormer E4 factorized LUT gate")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-json", type=Path, default=Path("results/qk_lutformer_e4_gate.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/QK_LUTFORMER_E4_GATE_REPORT.md"))
    args = parser.parse_args()
    paths = sorted(
        args.results_root.glob("qkformer_lut_e3_trainable_lut_*/metrics.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    runs = load_runs(paths)
    groups: Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        groups[(run["family"], run["checkpoint"], run["seed"])][run["mode"]] = run
    candidates = [evaluate_candidate(candidate, groups) for candidate in CANDIDATES]
    if any(candidate["pass"] for candidate in candidates):
        decision = "GO-METHOD"
    elif all(candidate["complete"] for candidate in candidates):
        decision = "GO-AUDIT"
    else:
        decision = "PENDING"
    report = {"decision": decision, "candidates": candidates, "num_input_runs": len(runs)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(f"decision={decision}")
    print(f"json={args.output_json}")
    print(f"markdown={args.output_md}")


if __name__ == "__main__":
    main()
