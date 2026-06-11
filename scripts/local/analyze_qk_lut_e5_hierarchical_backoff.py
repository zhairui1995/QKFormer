#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _pct(value: float | None) -> float | None:
    return None if value is None else 100.0 * float(value)


def _load_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(r"_e5_hier_calib(\d+)_seed(\d+)$")
    for path in paths:
        match = pattern.search(path.parent.name)
        if not match:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        overall = metrics["overall_reconstruction"]
        seed = int(match.group(2))
        calib_batches = int(match.group(1))
        fallback_sum = sum(
            float(overall.get(key, 0.0))
            for key in (
                "fallback_fraction_full",
                "fallback_fraction_plus_k",
                "fallback_fraction_plus_q_or_gate",
                "fallback_fraction_token_channel",
                "fallback_fraction_global",
            )
        )
        full_reduction = float(overall["component_full_address_relative_mse_reduction"])
        hierarchy_reduction = float(overall["hierarchical_backoff_relative_mse_reduction"])
        token_reduction = float(overall["component_token_channel_relative_mse_reduction"])
        row = {
            "result_dir": str(path.parent),
            "calibration_seed": seed,
            "calibration_batches": calib_batches,
            "eval_samples": int(overall["eval_samples"]),
            "global_mean_mse": float(overall["global_mean_mse"]),
            "token_channel_mse": float(overall["component_token_channel_mse"]),
            "plus_q_or_gate_mse": float(overall["component_plus_q_or_gate_mse"]),
            "plus_k_mse": float(overall["component_plus_k_mse"]),
            "full_address_mse": float(overall["component_full_address_mse"]),
            "hierarchical_backoff_mse": float(overall["hierarchical_backoff_mse"]),
            "shuffled_full_address_mse": float(overall["component_shuffled_full_address_mse"]),
            "token_channel_reduction_pct": _pct(token_reduction),
            "plus_q_or_gate_reduction_pct": _pct(overall["component_plus_q_or_gate_relative_mse_reduction"]),
            "plus_k_reduction_pct": _pct(overall["component_plus_k_relative_mse_reduction"]),
            "full_address_reduction_pct": _pct(full_reduction),
            "hierarchical_backoff_reduction_pct": _pct(hierarchy_reduction),
            "shuffled_full_address_reduction_pct": _pct(overall["component_shuffled_full_address_relative_mse_reduction"]),
            "hierarchical_backoff_hit_rate_pct": _pct(overall["hierarchical_backoff_hit_rate"]),
            "full_hit_rate_pct": _pct(overall["eval_address_hit_rate"]),
            "fallback_fraction_full": float(overall["fallback_fraction_full"]),
            "fallback_fraction_plus_k": float(overall["fallback_fraction_plus_k"]),
            "fallback_fraction_plus_q_or_gate": float(overall["fallback_fraction_plus_q_or_gate"]),
            "fallback_fraction_token_channel": float(overall["fallback_fraction_token_channel"]),
            "fallback_fraction_global": float(overall["fallback_fraction_global"]),
            "fallback_fraction_sum": fallback_sum,
            "hierarchical_supported_entries": int(overall["hierarchical_supported_entries"]),
            "hierarchical_nominal_entries": int(overall["hierarchical_nominal_entries"]),
            "monolithic_full_address_entries": int(overall["monolithic_full_address_entries"]),
            "full_address_supported_entries": int(overall["full_address_supported_entries"]),
            "hierarchical_supported_compression": float(overall["hierarchical_supported_compression"]),
            "hierarchical_nominal_compression": float(overall["hierarchical_nominal_compression"]),
            "hierarchy_beats_token": float(overall["hierarchical_backoff_mse"])
            < float(overall["component_token_channel_mse"]),
            "hierarchy_close_or_better_than_full": float(overall["hierarchical_backoff_mse"])
            <= float(overall["component_full_address_mse"]) * 1.001,
            "shuffled_worse_than_hierarchy": float(overall["component_shuffled_full_address_mse"])
            > float(overall["hierarchical_backoff_mse"]),
            "fallback_distribution_valid": abs(fallback_sum - 1.0) <= 1e-5,
            "hierarchy_retains_80pct_full_gain": (
                hierarchy_reduction / full_reduction >= 0.8 if full_reduction > 0 else False
            ),
            "hierarchy_uses_25pct_entries": float(overall["hierarchical_supported_compression"]) <= 0.25,
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (row["calibration_seed"], row["calibration_batches"]))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# E5 Hierarchical Backoff QK-LUT Report",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "| Seed | Calib | Hier. red. | Full red. | Token red. | Shuffled red. | Hier. hit | Eff. entries | Checks |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        check = "PASS" if row["row_pass_for_reconstruction"] else "FAIL"
        lines.append(
            "| {seed} | {calib} | {hier:.4f}% | {full:.4f}% | {token:.4f}% | "
            "{shuf:.4f}% | {hit:.4f}% | {entries} | {check} |".format(
                seed=row["calibration_seed"],
                calib=row["calibration_batches"],
                hier=row["hierarchical_backoff_reduction_pct"],
                full=row["full_address_reduction_pct"],
                token=row["token_channel_reduction_pct"],
                shuf=row["shuffled_full_address_reduction_pct"],
                hit=row["hierarchical_backoff_hit_rate_pct"],
                entries=row["hierarchical_supported_entries"],
                check=check,
            )
        )
    lines.extend(["", "## Gate Checks"])
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: {value}")
    lines.extend(["", "## Interpretation", report["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E5 hierarchical backoff QK-LUT reconstruction")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/qk_lutformer_e5_hierarchical_backoff.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/qk_lutformer_e5_hierarchical_backoff.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/qk_lutformer_e5_hierarchical_backoff_report.md"),
    )
    args = parser.parse_args()

    rows = _load_rows(args.results_root.glob("qkformer_lut_e1_recon_*_e5_hier_calib*_seed*/metrics.json"))
    for row in rows:
        row["row_pass_for_reconstruction"] = (
            row["hierarchy_beats_token"]
            and row["shuffled_worse_than_hierarchy"]
            and row["fallback_distribution_valid"]
        )

    sizes = sorted({int(row["calibration_batches"]) for row in rows})
    seeds = sorted({int(row["calibration_seed"]) for row in rows})
    rows_ge8 = [row for row in rows if int(row["calibration_batches"]) >= 8]
    rows_low = [row for row in rows if int(row["calibration_batches"]) in {1, 2, 4}]
    expected_complete = set(sizes) == {1, 2, 4, 8, 32, 128, 512} and set(seeds) == {42, 43, 44}
    checks = {
        "complete_21_runs": len(rows) == 21 and expected_complete,
        "hierarchy_beats_token_ge8_all": bool(rows_ge8) and all(row["hierarchy_beats_token"] for row in rows_ge8),
        "low_calib_close_or_better_than_full_all": bool(rows_low)
        and all(row["hierarchy_close_or_better_than_full"] for row in rows_low),
        "fallback_distribution_valid_all": bool(rows)
        and all(row["fallback_distribution_valid"] for row in rows),
        "shuffled_worse_than_hierarchy_all": bool(rows)
        and all(row["shuffled_worse_than_hierarchy"] for row in rows),
        "retains_80pct_full_gain_ge8_all": bool(rows_ge8)
        and all(row["hierarchy_retains_80pct_full_gain"] for row in rows_ge8),
        "uses_25pct_entries_ge8_all": bool(rows_ge8)
        and all(row["hierarchy_uses_25pct_entries"] for row in rows_ge8),
    }
    pass_checks = all(checks.values())
    partial_checks = (
        checks["complete_21_runs"]
        and checks["hierarchy_beats_token_ge8_all"]
        and checks["fallback_distribution_valid_all"]
        and checks["shuffled_worse_than_hierarchy_all"]
    )
    decision = "PASS" if pass_checks else "PARTIAL" if partial_checks else "FAIL"
    interpretation = {
        "PASS": (
            "Hierarchical backoff supports the method claim: it beats token/channel controls, "
            "keeps fallback behavior explicit, and meets the retained-gain/compression gate."
        ),
        "PARTIAL": (
            "Hierarchical backoff supports a scalable reconstruction audit, but at least one "
            "compression or low-calibration gate failed. Present it as a bounded method claim, "
            "not as a full solution to arbitrary Q/K LUT scaling."
        ),
        "FAIL": (
            "Hierarchical backoff does not yet support the proposed method claim. Keep the paper "
            "as GO-AUDIT and describe table explosion as an unresolved limitation."
        ),
    }[decision]
    report = {
        "decision": decision,
        "checks": checks,
        "num_rows": len(rows),
        "calibration_sizes": sizes,
        "calibration_seeds": seeds,
        "rows": rows,
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
