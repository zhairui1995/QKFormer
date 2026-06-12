#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _pct(value: float | None) -> float | None:
    return None if value is None else 100.0 * float(value)


def _load_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    pattern = re.compile(r"_e7_subspace_calib(\d+)_seed(\d+)_min(\d+)$")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        match = pattern.search(path.parent.name)
        if not match:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        overall = metrics["overall_reconstruction"]
        calib_batches = int(match.group(1))
        seed = int(match.group(2))
        min_count = int(match.group(3))
        token_red = float(overall["component_token_channel_relative_mse_reduction"])
        full_red = float(overall["component_full_address_relative_mse_reduction"])
        row = {
            "result_dir": str(path.parent),
            "calibration_batches": calib_batches,
            "calibration_seed": seed,
            "min_count": min_count,
            "eval_samples": int(overall["eval_samples"]),
            "global_mean_mse": float(overall["global_mean_mse"]),
            "token_channel_mse": float(overall["component_token_channel_mse"]),
            "full_address_mse": float(overall["component_full_address_mse"]),
            "shuffled_full_address_mse": float(overall["component_shuffled_full_address_mse"]),
            "subspace_tc_q_mse": float(overall["subspace_tc_q_mse"]),
            "subspace_tc_qk_mse": float(overall["subspace_tc_qk_mse"]),
            "shuffled_subspace_tc_q_mse": float(overall["shuffled_subspace_tc_q_mse"]),
            "shuffled_subspace_tc_qk_mse": float(overall["shuffled_subspace_tc_qk_mse"]),
            "token_channel_reduction_pct": _pct(token_red),
            "full_address_reduction_pct": _pct(full_red),
            "subspace_tc_q_reduction_pct": _pct(overall["subspace_tc_q_relative_mse_reduction"]),
            "subspace_tc_qk_reduction_pct": _pct(overall["subspace_tc_qk_relative_mse_reduction"]),
            "shuffled_subspace_tc_q_reduction_pct": _pct(
                overall["shuffled_subspace_tc_q_relative_mse_reduction"]
            ),
            "shuffled_subspace_tc_qk_reduction_pct": _pct(
                overall["shuffled_subspace_tc_qk_relative_mse_reduction"]
            ),
            "subspace_tc_q_supported_entries": int(overall["subspace_tc_q_supported_entries"]),
            "subspace_tc_qk_supported_entries": int(overall["subspace_tc_qk_supported_entries"]),
            "monolithic_full_address_entries": int(overall["monolithic_full_address_entries"]),
            "subspace_tc_q_supported_compression": float(overall["subspace_tc_q_supported_compression"]),
            "subspace_tc_qk_supported_compression": float(overall["subspace_tc_qk_supported_compression"]),
        }
        row["tc_q_beats_token"] = row["subspace_tc_q_mse"] < row["token_channel_mse"]
        row["tc_qk_beats_token"] = row["subspace_tc_qk_mse"] < row["token_channel_mse"]
        row["tc_q_shuffle_control"] = row["shuffled_subspace_tc_q_mse"] > row["subspace_tc_q_mse"]
        row["tc_qk_shuffle_control"] = row["shuffled_subspace_tc_qk_mse"] > row["subspace_tc_qk_mse"]
        row["tc_q_retention_vs_full"] = (
            float(overall["subspace_tc_q_relative_mse_reduction"]) / full_red
            if full_red > 0
            else 0.0
        )
        row["tc_qk_retention_vs_full"] = (
            float(overall["subspace_tc_qk_relative_mse_reduction"]) / full_red
            if full_red > 0
            else 0.0
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (row["calibration_seed"], row["calibration_batches"], row["min_count"]))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _best_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["calibration_seed"]), int(row["calibration_batches"]))].append(row)
    best: List[Dict[str, Any]] = []
    for items in grouped.values():
        candidates = sorted(
            items,
            key=lambda row: (
                not bool(row["tc_q_beats_token"] and row["tc_q_shuffle_control"]),
                not bool(row["tc_qk_beats_token"] and row["tc_qk_shuffle_control"]),
                -float(row["subspace_tc_q_reduction_pct"]),
                float(row["subspace_tc_q_supported_compression"]),
            ),
        )
        best.append(dict(candidates[0]))
    return sorted(best, key=lambda row: (row["calibration_seed"], row["calibration_batches"]))


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# E7 Subspace-Decoupled QK-LUT Reconstruction Report",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "| Seed | Calib | Min | Token red. | TC+Q red. | TC+QK red. | TC+Q comp. | TC+QK comp. | TC+Q pass | TC+QK pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["best_rows"]:
        lines.append(
            "| {seed} | {calib} | {min_count} | {token:.4f}% | {tcq:.4f}% | {tcqk:.4f}% | "
            "{tcq_comp:.4f} | {tcqk_comp:.4f} | {tcq_pass} | {tcqk_pass} |".format(
                seed=row["calibration_seed"],
                calib=row["calibration_batches"],
                min_count=row["min_count"],
                token=row["token_channel_reduction_pct"],
                tcq=row["subspace_tc_q_reduction_pct"],
                tcqk=row["subspace_tc_qk_reduction_pct"],
                tcq_comp=row["subspace_tc_q_supported_compression"],
                tcqk_comp=row["subspace_tc_qk_supported_compression"],
                tcq_pass=row["tc_q_beats_token"] and row["tc_q_shuffle_control"],
                tcqk_pass=row["tc_qk_beats_token"] and row["tc_qk_shuffle_control"],
            )
        )
    lines.extend(["", "## Gate Checks"])
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: {value}")
    lines.extend(["", "## Interpretation", report["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E7 subspace-decoupled QK-LUT reconstruction")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/qk_lutformer_e7_subspace.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/qk_lutformer_e7_subspace.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/qk_lutformer_e7_subspace_report.md"),
    )
    args = parser.parse_args()

    rows = _load_rows(args.results_root.glob("qkformer_lut_e1_recon_*_e7_subspace_calib*_seed*_min*/metrics.json"))
    best_rows = _best_rows(rows)
    expected_sizes = {8, 32, 128, 512}
    expected_seeds = {42, 43, 44}
    groups = {(int(row["calibration_seed"]), int(row["calibration_batches"])) for row in rows}
    expected_groups = {(seed, size) for seed in expected_seeds for size in expected_sizes}
    checks = {
        "has_expected_groups": expected_groups.issubset(groups),
        "tc_q_beats_token_all": bool(rows) and all(row["tc_q_beats_token"] for row in rows),
        "tc_q_shuffle_control_all": bool(rows) and all(row["tc_q_shuffle_control"] for row in rows),
        "tc_qk_beats_token_all": bool(rows) and all(row["tc_qk_beats_token"] for row in rows),
        "tc_qk_shuffle_control_all": bool(rows) and all(row["tc_qk_shuffle_control"] for row in rows),
    }
    if checks["has_expected_groups"] and checks["tc_q_beats_token_all"] and checks["tc_q_shuffle_control_all"]:
        decision = "PASS"
        interpretation = (
            "The compact TC+Q/G subspace residual LUT beats token/channel and its shuffled control "
            "across the planned calibration-size/seed grid."
        )
    elif checks["has_expected_groups"] and checks["tc_qk_beats_token_all"] and checks["tc_qk_shuffle_control_all"]:
        decision = "PARTIAL"
        interpretation = (
            "Subspace decomposition works only after adding K-specific sub-addresses. This supports "
            "a scalable decomposition direction, but not the tightest TC+Q/G compressed claim."
        )
    elif rows:
        decision = "PARTIAL"
        interpretation = (
            "Some subspace evidence exists, but the planned gate is incomplete or not stable. Treat "
            "E7 as an exploratory appendix result until the failed cells are understood."
        )
    else:
        decision = "FAIL"
        interpretation = "No E7 rows were found."

    report = {
        "decision": decision,
        "checks": checks,
        "rows": rows,
        "best_rows": best_rows,
        "calibration_sizes": sorted({int(row["calibration_batches"]) for row in rows}),
        "calibration_seeds": sorted({int(row["calibration_seed"]) for row in rows}),
        "min_counts": sorted({int(row["min_count"]) for row in rows}),
        "interpretation": interpretation,
    }

    _write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(f"decision={decision}")
    print(f"csv={args.output_csv}")
    print(f"json={args.output_json}")
    print(f"markdown={args.output_md}")


if __name__ == "__main__":
    main()
