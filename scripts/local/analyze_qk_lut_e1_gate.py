#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_t1_control_result(metrics: Dict[str, Any]) -> bool:
    overall = metrics.get("overall_reconstruction", {})
    return (
        int(metrics.get("model", {}).get("time_step", 0)) == 1
        and "token_channel_lut_mse" in overall
        and "shuffled_address_lut_mse" in overall
    )


def latest_by_seed(paths: Iterable[Path], family: str) -> List[Path]:
    selected: Dict[int, Path] = {}
    for path in paths:
        metrics = load_metrics(path)
        if metrics.get("model", {}).get("family") != family or not is_t1_control_result(metrics):
            continue
        seed = int(metrics.get("data", {}).get("calibration", {}).get("seed", -1))
        current = selected.get(seed)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            selected[seed] = path
    return [selected[seed] for seed in sorted(selected)]


def latest_by_checkpoint(paths: Iterable[Path], family: str) -> List[Path]:
    selected: Dict[str, Path] = {}
    for path in paths:
        metrics = load_metrics(path)
        if metrics.get("model", {}).get("family") != family or not is_t1_control_result(metrics):
            continue
        checkpoint = str(metrics.get("model", {}).get("checkpoint", {}).get("path") or "unknown")
        checkpoint_path = Path(checkpoint)
        checkpoint_id = "/".join(checkpoint_path.parts[-2:]) if checkpoint != "unknown" else checkpoint
        current = selected.get(checkpoint_id)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            selected[checkpoint_id] = path
    return [selected[key] for key in sorted(selected)]


def row(path: Path) -> Dict[str, Any]:
    metrics = load_metrics(path)
    overall = metrics["overall_reconstruction"]
    stages = metrics.get("per_stage_reconstruction", {})
    address_beats_global = overall["address_lut_mse"] < overall["global_mean_mse"]
    address_beats_token = overall["address_lut_mse"] < overall["token_channel_lut_mse"]
    shuffle_breaks_alignment = (
        overall["shuffled_address_lut_mse"] > overall["global_mean_mse"]
        and overall["shuffled_address_lut_mse"] > overall["address_lut_mse"]
    )
    return {
        "path": str(path),
        "family": metrics["model"]["family"],
        "checkpoint": "/".join(Path(str(metrics.get("model", {}).get("checkpoint", {}).get("path") or "unknown")).parts[-2:]),
        "calibration_seed": metrics.get("data", {}).get("calibration", {}).get("seed"),
        "overall": overall,
        "stages": stages,
        "checks": {
            "address_beats_global": address_beats_global,
            "address_beats_token_channel": address_beats_token,
            "shuffle_breaks_alignment": shuffle_breaks_alignment,
        },
    }


def markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# QK-LUTFormer E1 Gate Report",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "| Dataset | Checkpoint | Calibration seed | Address reduction | Token/channel reduction | Shuffled reduction | Checks |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in report["runs"]:
        overall = item["overall"]
        checks = item["checks"]
        check_text = ", ".join(key for key, value in checks.items() if not value) or "PASS"
        lines.append(
            f"| {item['family']} | {item['checkpoint']} | {item['calibration_seed']} | "
            f"{100 * overall['address_relative_mse_reduction']:.4f}% | "
            f"{100 * overall['token_channel_relative_mse_reduction']:.4f}% | "
            f"{100 * overall['shuffled_address_relative_mse_reduction']:.4f}% | {check_text} |"
        )
    lines.extend(["", "## Stage-level disclosure", ""])
    for item in report["runs"]:
        lines.append(f"### {item['family']} {item['checkpoint']} calibration seed {item['calibration_seed']}")
        lines.append("")
        lines.append("| Stage | Address reduction | Token/channel reduction | Shuffled reduction |")
        lines.append("|---|---:|---:|---:|")
        for stage, values in sorted(item["stages"].items()):
            lines.append(
                f"| {stage} | {100 * values['address_relative_mse_reduction']:.4f}% | "
                f"{100 * values['token_channel_relative_mse_reduction']:.4f}% | "
                f"{100 * values['shuffled_address_relative_mse_reduction']:.4f}% |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gate definition",
            "",
            "Gate passes only when three CIFAR-10 T=1 checkpoints and CIFAR-100 calibration seeds 42/43/44 are present, "
            "correct addresses beat the global mean and token/channel controls in every selected run, "
            "and shuffled addresses are worse than both correct addresses and the global mean.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the QK-LUTFormer E1 evidence gate")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-json", type=Path, default=Path("results/qk_lutformer_e1_gate.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/QK_LUTFORMER_E1_GATE_REPORT.md"))
    args = parser.parse_args()

    candidates = list(args.results_root.glob("qkformer_lut_e1_recon_*/metrics.json"))
    c10 = latest_by_checkpoint(candidates, "cifar10")
    c100 = latest_by_seed(candidates, "cifar100")
    c100_by_seed = {int(load_metrics(path)["data"]["calibration"]["seed"]): path for path in c100}
    selected = c10 + [c100_by_seed[seed] for seed in (42, 43, 44) if seed in c100_by_seed]
    runs = [row(path) for path in selected]
    complete = len(c10) >= 3 and all(seed in c100_by_seed for seed in (42, 43, 44))
    c10_rows = [item for item in runs if item["family"] == "cifar10"]
    c10_ok = len(c10_rows) >= 3 and all(all(item["checks"].values()) for item in c10_rows)
    c100_rows = [item for item in runs if item["family"] == "cifar100"]
    c100_ok = len(c100_rows) == 3 and all(all(item["checks"].values()) for item in c100_rows)
    report = {
        "decision": "PASS" if complete and c10_ok and c100_ok else "FAIL",
        "complete": complete,
        "runs": runs,
        "requirements": {"cifar10_three_checkpoints": c10_ok, "cifar100_three_calibration_seeds": c100_ok},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(f"decision={report['decision']}")
    print(f"json={args.output_json}")
    print(f"markdown={args.output_md}")


if __name__ == "__main__":
    main()
