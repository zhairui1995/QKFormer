#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]
FIG5_JSON = ROOT / "paper" / "figures" / "data" / "fig5_e7_component_ablation_data.json"
TCSLU_STATUS = ROOT / "paper" / "tables" / "data" / "tcslu_current_replacement_status.csv"
POWER_PROXY = ROOT / "results" / "qk_spike_lut_cifar100_param_power_proxy_20260618.json"
DATA = ROOT / "paper" / "tables" / "data"
OUT_CSV = DATA / "lookup_cost_proxy.csv"
OUT_JSON = DATA / "lookup_cost_proxy.json"
OUT_MD = ROOT / "paper" / "tables" / "lookup_cost_proxy.md"


def kib(byte_count: float) -> float:
    return byte_count / 1024.0


def ceil_log2(value: int) -> int:
    if value <= 1:
        return 1
    return math.ceil(math.log2(value))


def as_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sparse_bytes(entries: int, value_bytes: int, key_bytes: int = 4, count_bytes: int = 2) -> int:
    valid_bitset_bytes = math.ceil(entries / 8)
    return entries * (value_bytes + key_bytes + count_bytes) + valid_bitset_bytes


def dense_bytes(address_space: int, value_bytes: int) -> int:
    valid_bitset_bytes = math.ceil(address_space / 8)
    return address_space * value_bytes + valid_bitset_bytes


def make_cost_row(
    *,
    family: str,
    setting: str,
    table_name: str,
    supported_entries: int,
    value_kib: float,
    address_space: Optional[int],
    lookup_count_per_sample: Optional[int] = None,
    original_macs_per_sample: Optional[int] = None,
    source: str,
) -> Dict[str, str]:
    value_bytes = int(round(value_kib * 1024.0))
    sparse_fp32 = sparse_bytes(supported_entries, value_bytes=4)
    sparse_fp16 = sparse_bytes(supported_entries, value_bytes=2)
    row: Dict[str, str] = {
        "family": family,
        "setting": setting,
        "table_name": table_name,
        "supported_entries": str(supported_entries),
        "address_space": "" if address_space is None else str(address_space),
        "address_bits": "" if address_space is None else str(ceil_log2(address_space)),
        "prototype_only_fp32_kib": f"{kib(value_bytes):.4f}",
        "sparse_fp32_u32key_u16count_kib": f"{kib(sparse_fp32):.4f}",
        "sparse_fp16_u32key_u16count_kib": f"{kib(sparse_fp16):.4f}",
        "metadata_multiplier_vs_fp32_values": f"{sparse_fp32 / max(value_bytes, 1):.4f}",
        "dense_fp32_with_valid_bit_kib": "",
        "dense_fp16_with_valid_bit_kib": "",
        "dense_fp32_over_sparse_fp32": "",
        "lookup_count_per_sample": "" if lookup_count_per_sample is None else str(lookup_count_per_sample),
        "original_macs_per_sample": "" if original_macs_per_sample is None else str(original_macs_per_sample),
        "lookup_count_vs_original_macs": "",
        "source": source,
        "note": (
            "Analytical proxy only: sparse rows count FP32/FP16 prototype values, "
            "one uint32 key, one uint16 support counter, and one validity bit per "
            "supported entry; excludes alignment, cache behavior, control logic, "
            "allocator overhead, address-generation cost, and measured hardware cycles."
        ),
    }
    if address_space is not None:
        dense_fp32 = dense_bytes(address_space, value_bytes=4)
        dense_fp16 = dense_bytes(address_space, value_bytes=2)
        row["dense_fp32_with_valid_bit_kib"] = f"{kib(dense_fp32):.4f}"
        row["dense_fp16_with_valid_bit_kib"] = f"{kib(dense_fp16):.4f}"
        row["dense_fp32_over_sparse_fp32"] = f"{dense_fp32 / max(sparse_fp32, 1):.4f}"
    if lookup_count_per_sample and original_macs_per_sample:
        row["lookup_count_vs_original_macs"] = f"{lookup_count_per_sample / original_macs_per_sample:.6f}"
    return row


def e7_rows() -> List[Dict[str, str]]:
    payload = json.loads(FIG5_JSON.read_text(encoding="utf-8"))
    address_space = int(payload["source"]["compact_full_address_space_entries"])
    rows = []
    for item in payload["rows"]:
        rows.append(
            make_cost_row(
                family="E7 subspace reconstruction",
                setting="QKFormer CIFAR-100 T=1 seed42 calib128",
                table_name=item["label"],
                supported_entries=int(item["supported_entries"]),
                value_kib=float(item["idealized_fp32_kib"]),
                address_space=address_space,
                source=str(FIG5_JSON.relative_to(ROOT)),
            )
        )
    return rows


def power_proxy_by_t() -> Dict[str, Dict[str, float]]:
    if not POWER_PROXY.exists():
        return {}
    payload = json.loads(POWER_PROXY.read_text(encoding="utf-8"))
    return {str(int(row["time_step"])): row for row in payload.get("rows", [])}


def tcslu_rows() -> List[Dict[str, str]]:
    proxy = power_proxy_by_t()
    status_rows = read_csv(TCSLU_STATUS)
    selected_ids = {
        "qkf_c100_t1_tcslu": "TCSLU current replacement",
        "qkf_c100_t4_tcslu": "TCSLU current replacement",
        "spik_c10_t4_learned_temporal_channel": "learned temporal+channel boundary",
    }
    rows = []
    for row in status_rows:
        experiment_id = row["experiment_id"]
        if experiment_id not in selected_ids:
            continue
        memory = as_float(row["memory_kib"])
        supported = row["supported_entries"]
        if memory is None:
            continue
        entries = int(float(supported)) if supported else int(round(memory * 1024.0 / 4.0))
        address_space: Optional[int] = entries if row["architecture"] == "QKFormer" else None
        time_steps = row["time_steps"]
        p = proxy.get(time_steps, {}) if row["architecture"] == "QKFormer" else {}
        rows.append(
            make_cost_row(
                family="TCSLU current replacement",
                setting=f'{row["architecture"]} {row["dataset"]} T={time_steps}',
                table_name=selected_ids[experiment_id],
                supported_entries=entries,
                value_kib=memory,
                address_space=address_space,
                lookup_count_per_sample=(
                    int(p["output_scalars_or_lookups_per_sample"])
                    if "output_scalars_or_lookups_per_sample" in p
                    else None
                ),
                original_macs_per_sample=(
                    int(p["original_proj_macs_per_sample"])
                    if "original_proj_macs_per_sample" in p
                    else None
                ),
                source=row["source"],
            )
        )
    return rows


def markdown_table(rows: Iterable[Dict[str, str]], columns: List[str]) -> str:
    material = list(rows)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row.get(col, "") for col in columns) + " |" for row in material]
    return "\n".join([header, sep, *body]) + "\n"


def write_outputs(rows: List[Dict[str, str]]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(
        json.dumps(
            {
                "assumption": rows[0]["note"],
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    md = [
        "# Lookup Cost Proxy",
        "",
        "Generated by `scripts/local/build_lookup_cost_proxy_tables.py`.",
        "",
        (
            "This is an analytical accounting table, not a measured SRAM, "
            "latency, energy, or hardware-cycle result. The sparse metadata "
            "rows add one uint32 key, one uint16 support counter, and one "
            "validity bit per supported scalar entry."
        ),
        "",
        markdown_table(
            rows,
            [
                "family",
                "setting",
                "table_name",
                "supported_entries",
                "address_bits",
                "prototype_only_fp32_kib",
                "sparse_fp32_u32key_u16count_kib",
                "sparse_fp16_u32key_u16count_kib",
                "dense_fp32_with_valid_bit_kib",
                "lookup_count_vs_original_macs",
            ],
        ),
    ]
    OUT_MD.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    rows = e7_rows() + tcslu_rows()
    write_outputs(rows)
    print(
        json.dumps(
            {
                "csv": str(OUT_CSV.relative_to(ROOT)),
                "json": str(OUT_JSON.relative_to(ROOT)),
                "markdown": str(OUT_MD.relative_to(ROOT)),
                "rows": len(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
