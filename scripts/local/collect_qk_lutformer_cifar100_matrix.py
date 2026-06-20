#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    args = parser.parse_args()
    matrix_dir = args.matrix_dir.resolve()
    rows = []
    summaries = sorted(matrix_dir.glob("runs/*/summary.csv"))
    for summary in summaries:
        metadata_path = summary.parent / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        with summary.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                original_method = row.get("method", "")
                row["method"] = original_method.replace(
                    "qkformer_current", "qk_lutformer", 1
                )
                rows.append(
                    {
                        "case": metadata.get("case", ""),
                        "run_id": metadata.get("run_id", summary.parent.name),
                        "lut_mode": metadata.get("lut_mode", ""),
                        "calibration_seed": metadata.get("seed", ""),
                        "min_support": metadata.get("min_support", ""),
                        "corruption": metadata.get("corruption", "none"),
                        "corruption_severity": metadata.get("corruption_severity", 0),
                        "backbone": "QKFormer",
                        "method_family": "QK-LUTFormer/TCSLU",
                        "backbone_probe_method": original_method,
                        **row,
                        "source": str(summary.relative_to(matrix_dir)),
                    }
                )
    output = matrix_dir / "matrix_summary.csv"
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "experiment": "qk_lutformer_cifar100_tbd_matrix",
        "method": "QK-LUTFormer/TCSLU",
        "backbone": "QKFormer",
        "run_count": len(summaries),
        "row_count": len(rows),
        "summary_csv": str(output),
    }
    (matrix_dir / "matrix_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
