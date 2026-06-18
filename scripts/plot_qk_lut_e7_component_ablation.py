#!/usr/bin/env python3
"""Render the seed-42 E7 component and entry-budget ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
DATA_PATH = FIG_DIR / "data" / "fig5_e7_component_ablation_data.json"
CSV_PATH = FIG_DIR / "data" / "fig5_e7_component_ablation_data.csv"
OUTPUT_PATH = FIG_DIR / "fig5_e7_component_ablation.pdf"

COLORS = {
    "token": "#E69F00",
    "tc_q": "#56B4E9",
    "tc_qk": "#0072B2",
    "full": "#009E73",
    "shuffled": "#D55E00",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.6,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
        }
    )


def load_data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def write_csv(rows: list[dict]) -> None:
    fields = [
        "label",
        "reduction_pct",
        "shuffled_reduction_pct",
        "supported_entries",
        "entry_fraction_pct",
        "gain_retention_pct",
        "idealized_fp32_bytes",
        "idealized_fp32_kib",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render(payload: dict) -> None:
    rows = payload["rows"]
    labels = [row["label"].replace("/", "/\n") for row in rows]
    x = list(range(len(rows)))
    colors = [COLORS["token"], COLORS["tc_q"], COLORS["tc_qk"], COLORS["full"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(bottom=0.27, top=0.84)

    reductions = [row["reduction_pct"] for row in rows]
    ax1.bar(x, reductions, color=colors, width=0.68, label="Aligned")
    for idx, row in enumerate(rows):
        ax1.text(idx, row["reduction_pct"] + 0.09, f"{row['reduction_pct']:.2f}", ha="center", fontsize=6.8)
        shuffled = row["shuffled_reduction_pct"]
        if shuffled is not None:
            ax1.scatter(idx, shuffled, marker="x", s=28, linewidth=1.2, color=COLORS["shuffled"], zorder=3)
            ax1.text(idx, shuffled - 0.32, f"{shuffled:.2f}", ha="center", fontsize=6.5, color=COLORS["shuffled"])
    ax1.axhline(0, color="#333333", linewidth=0.65)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Relative MSE reduction (%)")
    ax1.set_ylim(-2.7, 5.15)
    ax1.grid(axis="y")
    ax1.set_title("(a) Aligned subspaces retain full-address gain", loc="left", fontweight="bold", fontsize=7.8)
    ax1.scatter([], [], marker="x", s=28, linewidth=1.2, color=COLORS["shuffled"], label="Shuffled subspace")
    ax1.legend(frameon=False, loc="lower right")

    fractions = [row["entry_fraction_pct"] for row in rows]
    ax2.bar(x, fractions, color=colors, width=0.68)
    for idx, row in enumerate(rows):
        label = f"{row['entry_fraction_pct']:.2f}%\n{row['idealized_fp32_kib']:.1f} KiB"
        ax2.text(idx, row["entry_fraction_pct"] + 0.75, label, ha="center", fontsize=6.6)
        if row["label"] in {"TC+Q/gate", "TC+QK"}:
            ax2.text(
                idx,
                row["entry_fraction_pct"] * 0.48,
                f"{row['gain_retention_pct']:.1f}% gain",
                ha="center",
                va="center",
                fontsize=6.4,
                color="#222222",
            )
    ax2.axhline(25.0, color=COLORS["shuffled"], linestyle="--", linewidth=0.9, label="25% entry budget")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Entries / compact address space (%)")
    ax2.set_ylim(0, 33.5)
    ax2.grid(axis="y")
    ax2.set_title("(b) Subspace entries stay below the 25% gate", loc="left", fontweight="bold", fontsize=7.8)
    ax2.legend(frameon=False, loc="upper left")

    fig.text(
        0.5,
        0.025,
        "CIFAR-100 T=1, seed 42, 128 calibration batches, full validation. "
        "KiB counts FP32 prototype values only.",
        ha="center",
        fontsize=7,
    )
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    payload = load_data()
    write_csv(payload["rows"])
    render(payload)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Wrote {CSV_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
