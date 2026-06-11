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
    pattern = re.compile(r"_e6_budget_calib(\d+)_seed(\d+)_min(\d+)$")
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
        fallback_keys = [
            "fallback_fraction_full",
            "fallback_fraction_plus_k",
            "fallback_fraction_plus_q_or_gate",
            "fallback_fraction_token_channel",
            "fallback_fraction_global",
        ]
        fallback_sum = sum(float(overall.get(key, 0.0)) for key in fallback_keys)
        full_reduction = float(overall["component_full_address_relative_mse_reduction"])
        hierarchy_reduction = float(overall["hierarchical_backoff_relative_mse_reduction"])
        token_reduction = float(overall["component_token_channel_relative_mse_reduction"])
        compression = float(overall["hierarchical_supported_compression"])
        row = {
            "result_dir": str(path.parent),
            "calibration_batches": calib_batches,
            "calibration_seed": seed,
            "min_count": min_count,
            "eval_samples": int(overall["eval_samples"]),
            "global_mean_mse": float(overall["global_mean_mse"]),
            "token_channel_mse": float(overall["component_token_channel_mse"]),
            "full_address_mse": float(overall["component_full_address_mse"]),
            "hierarchical_backoff_mse": float(overall["hierarchical_backoff_mse"]),
            "shuffled_full_address_mse": float(overall["component_shuffled_full_address_mse"]),
            "token_channel_reduction_pct": _pct(token_reduction),
            "full_address_reduction_pct": _pct(full_reduction),
            "hierarchical_backoff_reduction_pct": _pct(hierarchy_reduction),
            "shuffled_full_address_reduction_pct": _pct(
                overall["component_shuffled_full_address_relative_mse_reduction"]
            ),
            "hierarchical_backoff_hit_rate_pct": _pct(overall["hierarchical_backoff_hit_rate"]),
            "fallback_fraction_full": float(overall["fallback_fraction_full"]),
            "fallback_fraction_plus_k": float(overall["fallback_fraction_plus_k"]),
            "fallback_fraction_plus_q_or_gate": float(overall["fallback_fraction_plus_q_or_gate"]),
            "fallback_fraction_token_channel": float(overall["fallback_fraction_token_channel"]),
            "fallback_fraction_global": float(overall["fallback_fraction_global"]),
            "fallback_fraction_sum": fallback_sum,
            "hierarchical_supported_entries": int(overall["hierarchical_supported_entries"]),
            "monolithic_full_address_entries": int(overall["monolithic_full_address_entries"]),
            "hierarchical_supported_compression": compression,
            "hierarchy_beats_token": float(overall["hierarchical_backoff_mse"])
            < float(overall["component_token_channel_mse"]),
            "shuffled_worse_than_hierarchy": float(overall["component_shuffled_full_address_mse"])
            > float(overall["hierarchical_backoff_mse"]),
            "fallback_distribution_valid": abs(fallback_sum - 1.0) <= 1e-5,
            "retains_80pct_full_gain": (
                hierarchy_reduction / full_reduction >= 0.8 if full_reduction > 0 else False
            ),
            "uses_25pct_entries": compression <= 0.25,
        }
        row["budget_pass"] = (
            row["hierarchy_beats_token"]
            and row["shuffled_worse_than_hierarchy"]
            and row["fallback_distribution_valid"]
            and row["retains_80pct_full_gain"]
            and row["uses_25pct_entries"]
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
    for key, items in grouped.items():
        passing = [row for row in items if row["budget_pass"]]
        candidates = passing or items
        candidates = sorted(
            candidates,
            key=lambda row: (
                not row["uses_25pct_entries"],
                -float(row["hierarchical_backoff_reduction_pct"]),
                float(row["hierarchical_supported_compression"]),
                int(row["min_count"]),
            ),
        )
        chosen = dict(candidates[0])
        chosen["group_has_budget_pass"] = bool(passing)
        best.append(chosen)
    return sorted(best, key=lambda row: (row["calibration_seed"], row["calibration_batches"]))


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# E6 Budgeted Hierarchical Backoff QK-LUT Report",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "| Seed | Calib | Min count | Hier. red. | Token red. | Compression | Full fb | Global fb | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["best_rows"]:
        lines.append(
            "| {seed} | {calib} | {min_count} | {hier:.4f}% | {token:.4f}% | "
            "{comp:.4f} | {full_fb:.4f} | {global_fb:.4f} | {passed} |".format(
                seed=row["calibration_seed"],
                calib=row["calibration_batches"],
                min_count=row["min_count"],
                hier=row["hierarchical_backoff_reduction_pct"],
                token=row["token_channel_reduction_pct"],
                comp=row["hierarchical_supported_compression"],
                full_fb=row["fallback_fraction_full"],
                global_fb=row["fallback_fraction_global"],
                passed=row["group_has_budget_pass"],
            )
        )
    lines.extend(["", "## Gate Checks"])
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: {value}")
    lines.extend(["", "## Interpretation", report["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E6 budgeted hierarchical backoff QK-LUT")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/qk_lutformer_e6_budgeted_backoff.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/qk_lutformer_e6_budgeted_backoff.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/qk_lutformer_e6_budgeted_backoff_report.md"),
    )
    args = parser.parse_args()

    rows = _load_rows(args.results_root.glob("qkformer_lut_e1_recon_*_e6_budget_calib*_seed*_min*/metrics.json"))
    best_rows = _best_rows(rows)
    sizes = sorted({int(row["calibration_batches"]) for row in rows})
    seeds = sorted({int(row["calibration_seed"]) for row in rows})
    min_counts = sorted({int(row["min_count"]) for row in rows})
    expected_sizes = {8, 32, 128, 512}
    expected_seeds = {42, 43, 44}
    groups = {(int(row["calibration_seed"]), int(row["calibration_batches"])) for row in rows}
    expected_groups = {(seed, size) for seed in expected_seeds for size in expected_sizes}
    checks = {
        "has_expected_groups": expected_groups.issubset(groups),
        "has_budget_sweep": len(min_counts) >= 3,
        "fallback_distribution_valid_all": bool(rows) and all(row["fallback_distribution_valid"] for row in rows),
        "shuffled_worse_than_hierarchy_all": bool(rows) and all(row["shuffled_worse_than_hierarchy"] for row in rows),
        "budget_pass_each_expected_group": expected_groups.issubset(
            {
                (int(row["calibration_seed"]), int(row["calibration_batches"]))
                for row in rows
                if row["budget_pass"]
            }
        ),
    }
    decision = "PASS" if all(checks.values()) else "PARTIAL" if checks["has_expected_groups"] else "FAIL"
    interpretation = {
        "PASS": (
            "A support-threshold budgeted hierarchy meets the compact-table gate while retaining "
            "address reconstruction gains for every calibration size and seed."
        ),
        "PARTIAL": (
            "The budget sweep is complete, but at least one size/seed cannot satisfy the "
            "compression, retained-gain, and control gates simultaneously. Use as an appendix "
            "or as motivation for top-K/hash pruning rather than a main compression claim."
        ),
        "FAIL": (
            "The budget sweep is incomplete or fails core controls. Do not claim compressed "
            "hierarchical lookup yet."
        ),
    }[decision]
    report = {
        "decision": decision,
        "checks": checks,
        "num_rows": len(rows),
        "calibration_sizes": sizes,
        "calibration_seeds": seeds,
        "min_counts": min_counts,
        "rows": rows,
        "best_rows": best_rows,
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
