#!/usr/bin/env python3
"""Generate vector PDF figures for the QK-LUTFormer GO-AUDIT paper.

All quantitative panels read metrics from result JSON files. The two protocol
figures are generated as vector diagrams with matplotlib, not as raster
screenshots.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
DATA_DIR = FIG_DIR / "data"

FIG3_RUNS = [
    ("C10 ckpt42", "cifar10", "ckpt42", ROOT / "results/qkformer_lut_e1_recon_20260611_151336_c10_t1_ckpt42_controls/metrics.json"),
    ("C10 ckpt43", "cifar10", "ckpt43", ROOT / "results/qkformer_lut_e1_recon_20260611_150653_c10_t1_ckpt43_controls/metrics.json"),
    ("C10 ckpt44", "cifar10", "ckpt44", ROOT / "results/qkformer_lut_e1_recon_20260611_150654_c10_t1_ckpt44_controls/metrics.json"),
    ("C100 seed42", "cifar100", "calib42", ROOT / "results/qkformer_lut_e1_recon_20260611_145102_c100_stability_seed42/metrics.json"),
    ("C100 seed43", "cifar100", "calib43", ROOT / "results/qkformer_lut_e1_recon_20260611_145103_c100_stability_seed43/metrics.json"),
    ("C100 seed44", "cifar100", "calib44", ROOT / "results/qkformer_lut_e1_recon_20260611_145104_c100_stability_seed44/metrics.json"),
]

FIG4_RUNS = {
    8: [
        ROOT / "results/qkformer_lut_e1_recon_20260611_152044_c100_calib8_controls/metrics.json",
        ROOT / "results/qkformer_lut_e1_recon_20260611_175354_c100_calib8_seed43_controls/metrics.json",
        ROOT / "results/qkformer_lut_e1_recon_20260611_180128_c100_calib8_seed44_controls/metrics.json",
    ],
    32: [
        ROOT / "results/qkformer_lut_e1_recon_20260611_152045_c100_calib32_controls/metrics.json",
        ROOT / "results/qkformer_lut_e1_recon_20260611_175354_c100_calib32_seed43_controls/metrics.json",
        ROOT / "results/qkformer_lut_e1_recon_20260611_180128_c100_calib32_seed44_controls/metrics.json",
    ],
    128: [
        ROOT / "results/qkformer_lut_e1_recon_20260611_145102_c100_stability_seed42/metrics.json",
        ROOT / "results/qkformer_lut_e1_recon_20260611_145103_c100_stability_seed43/metrics.json",
        ROOT / "results/qkformer_lut_e1_recon_20260611_145104_c100_stability_seed44/metrics.json",
    ],
    512: [
        ROOT / "results/qkformer_lut_e1_recon_20260611_152046_c100_calib512_controls/metrics.json",
        ROOT / "results/qkformer_lut_e1_recon_20260611_175354_c100_calib512_seed43_controls/metrics.json",
        ROOT / "results/qkformer_lut_e1_recon_20260611_180128_c100_calib512_seed44_controls/metrics.json",
    ],
}

PALETTE = {
    "address": "#0072B2",
    "token": "#E69F00",
    "shuffled": "#D55E00",
    "global": "#666666",
    "green": "#009E73",
    "purple": "#CC79A7",
    "light_blue": "#D9EAF7",
    "light_orange": "#FBE6C3",
    "light_red": "#F8D4C7",
    "light_gray": "#F2F2F2",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.6,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.5,
    })


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> float:
    return 100.0 * float(value)


def draw_box(ax, xy, wh, text, fc="#FFFFFF", ec="#333333", lw=0.8, fontsize=8, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return patch


def draw_arrow(ax, start, end, color="#333333", lw=0.9, style="-|>"):
    arr = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=8, linewidth=lw, color=color)
    ax.add_patch(arr)
    return arr


def draw_plain_box(ax, xy, wh, text, fc="#FFFFFF", ec="#333333", lw=0.75, fontsize=7.5, weight="normal"):
    x, y = xy
    w, h = wh
    patch = Rectangle((x, y), w, h, linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return patch


def panel_label(ax, label, x=0.01, y=0.98):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, weight="bold", clip_on=False)


def save_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def figure1_audit_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.03, 0.96, "Q/K lookup-addressability audit", fontsize=10, weight="bold", ha="left")
    ax.text(0.03, 0.915, "The paper separates address structure from downstream utility.", fontsize=7.6, color="#333333")

    # Main evidence path.
    lane_y = 0.64
    lane_h = 0.18
    stage_w = 0.18
    stage_gap = 0.055
    xs = [0.035, 0.035 + stage_w + stage_gap, 0.035 + 2 * (stage_w + stage_gap), 0.035 + 3 * (stage_w + stage_gap)]
    stages = [
        ("Trace", "binary Q/K spikes\n+ token/channel state", PALETTE["light_blue"]),
        ("Pack", "deterministic\nlookup address", "#E8F4EC"),
        ("Calibrate", "address-indexed\nresponse prototype", "#FFF4D6"),
        ("Evaluate", "held-out local\nresponse MSE", "#F3E8F5"),
    ]
    for x, (title, body, color) in zip(xs, stages):
        draw_plain_box(ax, (x, lane_y), (stage_w, lane_h), "", fc=color, ec="#333333", lw=0.75)
        ax.text(x + 0.012, lane_y + lane_h - 0.035, title.upper(), fontsize=6.6, weight="bold", ha="left", va="top", color="#333333")
        ax.text(x + stage_w / 2, lane_y + 0.065, body, fontsize=7.4, ha="center", va="center")
    for i in range(3):
        draw_arrow(ax, (xs[i] + stage_w + 0.006, lane_y + lane_h / 2), (xs[i + 1] - 0.008, lane_y + lane_h / 2), lw=0.8)

    # Controls as protocol row.
    ax.text(0.035, 0.515, "Negative controls used at the reconstruction gate", fontsize=8.6, weight="bold", ha="left")
    ctrl_y = 0.315
    ctrl_w = 0.205
    ctrl_gap = 0.03
    controls = [
        ("Global mean", "removes address\ninformation", PALETTE["light_gray"], PALETTE["global"]),
        ("Token/channel", "keeps coarse\ncontext only", PALETTE["light_orange"], PALETTE["token"]),
        ("Correct Q/K", "uses aligned\nbinary address", PALETTE["light_blue"], PALETTE["address"]),
        ("Shuffled Q/K", "same table,\nbroken alignment", PALETTE["light_red"], PALETTE["shuffled"]),
    ]
    for i, (title, body, color, edge) in enumerate(controls):
        x = 0.035 + i * (ctrl_w + ctrl_gap)
        draw_plain_box(ax, (x, ctrl_y), (ctrl_w, 0.145), "", fc=color, ec=edge, lw=0.9)
        ax.text(x + 0.012, ctrl_y + 0.112, title, fontsize=7.4, weight="bold", ha="left", va="center", color="#222222")
        ax.text(x + ctrl_w / 2, ctrl_y + 0.047, body, fontsize=7.0, ha="center", va="center")

    # Claim boundary.
    ax.text(0.035, 0.205, "Claim boundary", fontsize=8.6, weight="bold", ha="left")
    draw_plain_box(
        ax,
        (0.035, 0.055),
        (0.43, 0.115),
        "Supported\naddress-specific response reconstruction",
        fc="#F7FBFF",
        ec=PALETTE["address"],
        lw=0.9,
        fontsize=7.2,
    )
    draw_plain_box(
        ax,
        (0.535, 0.055),
        (0.43, 0.115),
        "Not claimed\nstable accuracy, hardware gain, full wrapper",
        fc="#FFF8F4",
        ec=PALETTE["shuffled"],
        lw=0.9,
        fontsize=7.2,
    )
    draw_arrow(ax, (0.245, ctrl_y), (0.245, 0.173), color="#555555", lw=0.65)
    draw_arrow(ax, (0.750, ctrl_y), (0.750, 0.173), color="#555555", lw=0.65)

    fig.savefig(FIG_DIR / "fig1_audit_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def figure2_address_controls() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15))
    titles = ["Correct address alignment", "Shuffled address control", "Coarse / global controls"]
    for ax, title in zip(axes, titles):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.04, 0.90, title, ha="left", va="top", fontsize=8.0, weight="bold")

    panel_label(axes[0], "(a)", x=0.00, y=1.03)
    panel_label(axes[1], "(b)", x=0.00, y=1.03)
    panel_label(axes[2], "(c)", x=0.00, y=1.03)

    # Correct address: identity mapping from observed addresses to prototypes.
    ax = axes[0]
    for i, y in enumerate([0.70, 0.51, 0.32]):
        draw_plain_box(ax, (0.07, y - 0.045), (0.28, 0.09), f"A{i+1}: Q/K bits", fc=PALETTE["light_blue"], fontsize=6.8)
        draw_plain_box(ax, (0.64, y - 0.045), (0.28, 0.09), f"P{i+1}: response", fc="#FFF4D6", fontsize=6.8)
        draw_arrow(ax, (0.36, y), (0.63, y), color=PALETTE["address"], lw=1.1)
    ax.text(0.07, 0.15, "Same address retrieves its\ncalibrated response prototype.", fontsize=6.8)

    # Shuffled address: crossed mapping.
    ax = axes[1]
    left_y = [0.70, 0.51, 0.32]
    right_y = [0.32, 0.70, 0.51]
    for i, y in enumerate(left_y):
        draw_plain_box(ax, (0.08, y - 0.045), (0.27, 0.09), f"A{i+1}", fc=PALETTE["light_blue"], fontsize=6.8)
    for i, y in enumerate([0.70, 0.51, 0.32]):
        draw_plain_box(ax, (0.64, y - 0.045), (0.27, 0.09), f"P{i+1}", fc="#FFF4D6", fontsize=6.8)
    for y0, y1 in zip(left_y, right_y):
        draw_arrow(ax, (0.36, y0), (0.63, y1), color=PALETTE["shuffled"], lw=1.0)
    ax.text(0.08, 0.15, "Same table capacity, but the\naddress--prototype pairing is broken.", fontsize=6.8)

    # Coarse/global controls.
    ax = axes[2]
    draw_plain_box(ax, (0.06, 0.61), (0.30, 0.11), "Token/channel\nbin", fc=PALETTE["light_orange"], fontsize=6.8)
    draw_plain_box(ax, (0.06, 0.38), (0.30, 0.11), "Candidate /\nbackground", fc=PALETTE["light_gray"], fontsize=6.8)
    draw_plain_box(ax, (0.06, 0.16), (0.30, 0.11), "Global\nmean", fc="#FFFFFF", fontsize=6.8)
    draw_plain_box(ax, (0.62, 0.47), (0.31, 0.14), "Coarse response\nbaseline", fc="#FFF4D6", fontsize=6.8)
    for y0 in [0.68, 0.43, 0.20]:
        draw_arrow(ax, (0.35, y0), (0.61, 0.55), color="#666666", lw=0.85)
    ax.text(0.06, 0.76, "Controls remove fine Q/K detail\nor remove address dependence.", fontsize=6.8)

    fig.savefig(FIG_DIR / "fig2_address_controls.pdf", bbox_inches="tight")
    plt.close(fig)


def read_e1_row(label: str, dataset: str, split_id: str, path: Path) -> Dict[str, Any]:
    metrics = load_metrics(path)
    overall = metrics["overall_reconstruction"]
    data = metrics.get("data", {})
    row = {
        "label": label,
        "dataset": dataset,
        "split_id": split_id,
        "source_metrics": str(path.relative_to(ROOT)),
        "calibration_batches": data.get("calibration", {}).get("num_batches"),
        "calibration_seed": data.get("calibration", {}).get("seed"),
        "evaluation_batches": data.get("evaluation", {}).get("num_batches"),
        "address_reduction_pct": pct(overall["address_relative_mse_reduction"]),
        "token_channel_reduction_pct": pct(overall["token_channel_relative_mse_reduction"]),
        "shuffled_reduction_pct": pct(overall["shuffled_address_relative_mse_reduction"]),
        "candidate_background_reduction_pct": pct(overall["candidate_background_relative_mse_reduction"]),
        "hit_rate_pct": pct(overall["eval_address_hit_rate"]),
        "global_mse": overall["global_mean_mse"],
        "address_mse": overall["address_lut_mse"],
        "token_channel_mse": overall["token_channel_lut_mse"],
        "shuffled_mse": overall["shuffled_address_lut_mse"],
    }
    return row


def collect_fig3_data() -> List[Dict[str, Any]]:
    return [read_e1_row(label, dataset, split_id, path) for label, dataset, split_id, path in FIG3_RUNS]


def collect_fig4_data() -> List[Dict[str, Any]]:
    rows = []
    for size, paths in FIG4_RUNS.items():
        seed_rows = [read_e1_row(f"{size} batches", "cifar100", f"calib{size}", path) for path in paths]
        row = {
            "label": f"{size} batches",
            "dataset": "cifar100",
            "split_id": f"calib{size}",
            "calibration_size": size,
            "calibration_seed": "42,43,44",
            "num_seeds": len(seed_rows),
            "evaluation_batches": 0,
            "source_metrics": ";".join(r["source_metrics"] for r in seed_rows),
        }
        for key in (
            "address_reduction_pct",
            "token_channel_reduction_pct",
            "shuffled_reduction_pct",
            "candidate_background_reduction_pct",
            "hit_rate_pct",
            "global_mse",
            "address_mse",
            "token_channel_mse",
            "shuffled_mse",
        ):
            values = [float(r[key]) for r in seed_rows]
            row[key] = statistics.mean(values)
            row[f"{key}_std"] = statistics.pstdev(values)
        rows.append(row)
    rows.sort(key=lambda r: int(r["calibration_size"]))
    return rows


def figure3_e1_reconstruction(rows: List[Dict[str, Any]]) -> None:
    labels = [r["label"].replace(" ", "\n") for r in rows]
    x = list(range(len(rows)))
    address = [r["address_reduction_pct"] for r in rows]
    token = [r["token_channel_reduction_pct"] for r in rows]
    shuffled = [r["shuffled_reduction_pct"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.15), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.15], "hspace": 0.08})
    width = 0.34
    ax1.bar([i - width / 2 for i in x], address, width=width, color=PALETTE["address"], label="Correct Q/K address")
    ax1.bar([i + width / 2 for i in x], token, width=width, color=PALETTE["token"], label="Token/channel")
    for i, (a, t) in enumerate(zip(address, token)):
        ax1.plot([i - width / 2, i + width / 2], [a, t], color="#333333", lw=0.45, alpha=0.65)
    ax1.axhline(0, color="#333333", lw=0.6)
    ax1.set_ylabel("Relative MSE\nreduction (%)")
    ax1.set_ylim(0, max(address) * 1.26)
    ax1.grid(axis="y", linestyle="-", alpha=0.9)
    ax1.legend(frameon=False, loc="upper right", ncol=2)
    panel_label(ax1, "(a)", x=0.0, y=1.04)
    ax1.set_title("Correct Q/K addresses consistently beat coarse token/channel controls", pad=5)
    ax1.axvline(2.5, color="#AAAAAA", lw=0.65)
    ax1.text(1.0, ax1.get_ylim()[1] * 0.91, "CIFAR-10 checkpoints", ha="center", va="center", fontsize=7.2, color="#333333")
    ax1.text(4.0, ax1.get_ylim()[1] * 0.91, "CIFAR-100 calibration seeds", ha="center", va="center", fontsize=7.2, color="#333333")

    ax2.bar(x, shuffled, width=0.46, color=PALETTE["shuffled"], label="Shuffled address")
    ax2.axhline(0, color="#333333", lw=0.6)
    ax2.set_ylabel("Shuffled\ncontrol (%)")
    ax2.set_ylim(min(shuffled) * 1.15, 2.0)
    ax2.grid(axis="y", linestyle="-", alpha=0.9)
    panel_label(ax2, "(b)", x=0.0, y=1.05)
    ax2.axvline(2.5, color="#AAAAAA", lw=0.65)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.text(0.01, 0.08, "Negative values mean MSE is worse than the global mean baseline.", transform=ax2.transAxes, fontsize=7)

    fig.savefig(FIG_DIR / "fig3_e1_reconstruction.pdf", bbox_inches="tight")
    plt.close(fig)


def figure4_calibration_sweep(rows: List[Dict[str, Any]]) -> None:
    sizes = [int(r["calibration_size"]) for r in rows]
    address = [r["address_reduction_pct"] for r in rows]
    token = [r["token_channel_reduction_pct"] for r in rows]
    hit = [r["hit_rate_pct"] for r in rows]
    address_std = [r["address_reduction_pct_std"] for r in rows]
    token_std = [r["token_channel_reduction_pct_std"] for r in rows]
    hit_std = [r["hit_rate_pct_std"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax1.errorbar(sizes, address, yerr=address_std, marker="o", color=PALETTE["address"], lw=1.4, capsize=2, label="Correct Q/K address")
    ax1.errorbar(sizes, token, yerr=token_std, marker="s", color=PALETTE["token"], lw=1.4, capsize=2, label="Token/channel")
    ax1.fill_between(sizes, token, address, color=PALETTE["address"], alpha=0.10, linewidth=0)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sizes)
    ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax1.set_xlabel("Calibration batches")
    ax1.set_ylabel("Relative MSE reduction (%)")
    ax1.set_ylim(3.0, max(address) + 0.35)
    ax1.grid(True, axis="both", alpha=0.9)
    ax1.legend(frameon=False, loc="lower right")
    panel_label(ax1, "(a)", x=0.0, y=1.04)
    ax1.annotate("8 batches recover\n~85% of 512-batch gain", xy=(8, address[0]), xytext=(10, address[0] + 0.33),
                 arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#555555"}, fontsize=6.8)

    ax2.errorbar(sizes, hit, yerr=hit_std, marker="o", color=PALETTE["green"], lw=1.4, capsize=2)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sizes)
    ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax2.set_xlabel("Calibration batches")
    ax2.set_ylabel("Validation hit rate (%)")
    ax2.set_ylim(min(hit) - 0.08, 100.01)
    ax2.grid(True, axis="both", alpha=0.9)
    panel_label(ax2, "(b)", x=0.0, y=1.04)
    ax2.text(0.02, 0.08, "Mean over calibration seeds 42--44", transform=ax2.transAxes, fontsize=7)

    fig.suptitle("CIFAR-100 calibration-size sweep: correct address remains above coarse control", y=1.04, fontsize=9)
    fig.savefig(FIG_DIR / "fig4_calibration_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def write_data_snapshots(fig3_rows: List[Dict[str, Any]], fig4_rows: List[Dict[str, Any]]) -> None:
    fig3_fields = [
        "label", "dataset", "split_id", "calibration_batches", "calibration_seed",
        "evaluation_batches", "address_reduction_pct", "token_channel_reduction_pct",
        "shuffled_reduction_pct", "candidate_background_reduction_pct", "hit_rate_pct",
        "global_mse", "address_mse", "token_channel_mse", "shuffled_mse", "source_metrics",
    ]
    fig4_fields = [
        "calibration_size", "dataset", "calibration_seed", "num_seeds", "evaluation_batches",
        "address_reduction_pct", "address_reduction_pct_std",
        "token_channel_reduction_pct", "token_channel_reduction_pct_std",
        "shuffled_reduction_pct", "candidate_background_reduction_pct",
        "hit_rate_pct", "hit_rate_pct_std", "global_mse",
        "address_mse", "token_channel_mse", "shuffled_mse", "source_metrics",
    ]
    save_csv(DATA_DIR / "fig3_e1_reconstruction_data.csv", fig3_rows, fig3_fields)
    save_json(DATA_DIR / "fig3_e1_reconstruction_data.json", fig3_rows)
    save_csv(DATA_DIR / "fig4_calibration_sweep_data.csv", fig4_rows, fig4_fields)
    save_json(DATA_DIR / "fig4_calibration_sweep_data.json", fig4_rows)


def main() -> None:
    configure_matplotlib()
    ensure_dirs()
    fig3_rows = collect_fig3_data()
    fig4_rows = collect_fig4_data()
    write_data_snapshots(fig3_rows, fig4_rows)
    figure1_audit_overview()
    figure2_address_controls()
    figure3_e1_reconstruction(fig3_rows)
    figure4_calibration_sweep(fig4_rows)
    print("Wrote:")
    for path in [
        FIG_DIR / "fig1_audit_overview.pdf",
        FIG_DIR / "fig2_address_controls.pdf",
        FIG_DIR / "fig3_e1_reconstruction.pdf",
        FIG_DIR / "fig4_calibration_sweep.pdf",
        DATA_DIR / "fig3_e1_reconstruction_data.csv",
        DATA_DIR / "fig4_calibration_sweep_data.csv",
    ]:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
