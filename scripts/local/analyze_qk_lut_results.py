#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: (path.stat().st_mtime, str(path)))


def latest_t1_train(results: Path) -> Path | None:
    candidates: list[Path] = []
    for path in results.glob("qkformer_cifar10_train_*"):
        manifest = path / "checkpoint_manifest.json"
        if not manifest.exists():
            continue
        data = load_json(manifest)
        if str(data.get("time_step")) == "1":
            candidates.append(path)
    return latest(candidates)


def latest_t1_e0(results: Path) -> Path | None:
    candidates: list[Path] = []
    for metrics in results.glob("qkformer_lut_e0_diag_*/metrics.json"):
        data = load_json(metrics)
        if str(data.get("model", {}).get("time_step")) == "1":
            candidates.append(metrics.parent)
    return latest(candidates)


def latest_t1_e1(results: Path) -> Path | None:
    candidates: list[Path] = []
    for metrics in results.glob("qkformer_lut_e1_recon_*/metrics.json"):
        data = load_json(metrics)
        if str(data.get("model", {}).get("time_step")) == "1":
            candidates.append(metrics.parent)
    return latest(candidates)


def latest_t1_e3(results: Path, limit: int, checkpoint_marker: str | None = None) -> list[Path]:
    candidates_by_key: dict[tuple[str, str, str, str, str], Path] = {}
    for metrics in results.glob("qkformer_lut_e3_trainable_lut_*/metrics.json"):
        data = load_json(metrics)
        if str(data.get("model", {}).get("time_step")) == "1":
            checkpoint_path = str(data.get("model", {}).get("checkpoint", {}).get("path") or "")
            if checkpoint_marker and checkpoint_marker not in checkpoint_path:
                continue
            mode = data.get("adapter_summary", {}).get("mode") or data.get("adapter_config", {}).get("mode", "unknown")
            seed = (
                data.get("env_overrides", {}).get("QKFORMER_LUT_E3_SEED")
                or data.get("adapter_config", {}).get("mode_seed")
                or data.get("experiment", {}).get("seed")
            )
            alpha = data.get("env_overrides", {}).get("QKFORMER_LUT_E3_ALPHA_INIT")
            if alpha is None:
                alpha = data.get("adapter_config", {}).get("alpha_init")
            address_scale = data.get("env_overrides", {}).get("QKFORMER_LUT_E3_ADDRESS_SCALE")
            if address_scale is None:
                address_scale = data.get("adapter_config", {}).get("address_scale")
            key = (checkpoint_path, str(mode), str(seed), str(alpha), str(address_scale))
            old = candidates_by_key.get(key)
            if old is None or metrics.parent.stat().st_mtime > old.stat().st_mtime:
                candidates_by_key[key] = metrics.parent
    candidates = list(candidates_by_key.values())
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)[:limit]


def best_summary_row(train_dir: Path) -> tuple[str | None, str | None, str | None]:
    summaries = list(train_dir.glob("**/summary.csv"))
    if not summaries:
        return None, None, None
    rows = list(csv.DictReader(summaries[0].open(newline="", encoding="utf-8")))
    if not rows:
        return None, None, None
    best = max(rows, key=lambda row: float(row.get("eval_top1") or -1))
    last = rows[-1]
    return best.get("epoch"), best.get("eval_top1"), last.get("eval_top1")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize_rows(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for mode, rows in sorted(grouped.items()):
        deltas = [float(row["delta_top1"]) for row in rows if row["delta_top1"] is not None]
        losses = [float(row["delta_loss"]) for row in rows if row["delta_loss"] is not None]
        kls = [float(row["kl"]) for row in rows if row["kl"] is not None]
        logit_mses = [float(row["logit_mse"]) for row in rows if row["logit_mse"] is not None]
        local_mses = [float(row["local_mse"]) for row in rows if row["local_mse"] is not None]
        baseline_top1 = [float(row["baseline_top1"]) for row in rows if row["baseline_top1"] is not None]
        replacement_top1 = [float(row["replacement_top1"]) for row in rows if row["replacement_top1"] is not None]
        if not deltas:
            continue
        summary[mode] = {
            "n": len(deltas),
            "mean_baseline_top1": mean_or_none(baseline_top1),
            "mean_replacement_top1": mean_or_none(replacement_top1),
            "mean_delta_top1": statistics.mean(deltas),
            "min_delta_top1": min(deltas),
            "max_delta_top1": max(deltas),
            "mean_delta_loss": mean_or_none(losses),
            "mean_kl": mean_or_none(kls),
            "mean_logit_mse": mean_or_none(logit_mses),
            "mean_local_mse": mean_or_none(local_mses),
        }
    return summary


def decide_verdict(summary: dict[str, dict[str, Any]]) -> str:
    required = {"address_lut", "global_mean", "token_channel_lut"}
    if not required.issubset(summary):
        return "PENDING"
    address = float(summary["address_lut"]["mean_delta_top1"])
    controls = [
        float(summary["global_mean"]["mean_delta_top1"]),
        float(summary["token_channel_lut"]["mean_delta_top1"]),
    ]
    if "shuffled_address_lut" in summary:
        controls.append(float(summary["shuffled_address_lut"]["mean_delta_top1"]))
    best_control = max(controls)
    if address > 0 and address > best_control:
        return "CONDITIONAL GO: address_lut leads T=1 controls; repeat on another checkpoint before broad paper claim"
    if address > best_control:
        return "CONDITIONAL GO: address_lut beats controls but mean gain is not positive"
    return "NO-GO for address-specific accuracy claim at this setting"


def write_markdown_report(
    path: Path,
    root: Path,
    train_dir: Path | None,
    train_best: tuple[str | None, str | None, str | None] | None,
    e0_dir: Path | None,
    e0: dict[str, Any] | None,
    grouped: dict[str, list[dict[str, Any]]],
    summary: dict[str, dict[str, Any]],
    verdict: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# QK-LUTFormer T=1 E3 5-Way Control Results")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Setting: CIFAR-10 T=1, conservative E3 residual LUT adapter, stage1-only target.")
    lines.append("- Five-way comparison: QKFormer baseline plus `address_lut`, `global_mean`, `token_channel_lut`, and `shuffled_address_lut` controls.")
    lines.append("- Artifact policy: this report is based on lightweight `metrics.json`, `train_log.txt`, manifest, and summary files only; checkpoint weights are not required locally.")
    lines.append("")
    lines.append("## Backbone And E0 Context")
    lines.append("")
    if train_dir and train_best:
        epoch, best_top1, last_top1 = train_best
        lines.append(f"- T=1 train result: `{train_dir.name}`.")
        lines.append(f"- Best validation Acc@1: {best_top1}% at epoch {epoch}; final logged Acc@1: {last_top1}%.")
    else:
        lines.append("- T=1 train result: missing from local lightweight artifacts.")
    if e0_dir and e0:
        lines.append(f"- T=1 E0 result: `{e0_dir.name}`.")
        lines.append(
            "- E0 address coverage / singleton fraction / conditional variance: "
            f"{fmt(e0.get('address_coverage'), 6)} / {fmt(e0.get('singleton_fraction'), 6)} / "
            f"{fmt(e0.get('conditional_variance'), 6)}."
        )
    else:
        lines.append("- T=1 E0 result: missing from local lightweight artifacts.")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| Setting | n | Baseline Acc@1 | Adapter Acc@1 | Delta Acc@1 | Delta Loss | KL | Logit MSE | Local MSE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    fixed_order = ["address_lut", "global_mean", "token_channel_lut", "shuffled_address_lut"]
    ordered_modes = [mode for mode in fixed_order if mode in summary]
    ordered_modes.extend(mode for mode in sorted(summary) if mode not in ordered_modes)
    for mode in ordered_modes:
        item = summary.get(mode)
        lines.append(
            f"| `{mode}` | {item['n']} | {fmt(item.get('mean_baseline_top1'), 4)} | "
            f"{fmt(item.get('mean_replacement_top1'), 4)} | {fmt(item.get('mean_delta_top1'), 4)} | "
            f"{fmt(item.get('mean_delta_loss'), 6)} | {fmt(item.get('mean_kl'), 6)} | "
            f"{fmt(item.get('mean_logit_mse'), 6)} | {fmt(item.get('mean_local_mse'), 6)} |"
        )
    lines.append("")
    lines.append("## Per-Seed Rows")
    lines.append("")
    lines.append("| Mode | Seed | Baseline Acc@1 | Adapter Acc@1 | Delta Acc@1 | Delta Loss | Result Dir |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for mode in ordered_modes:
        for row in sorted(grouped.get(mode, []), key=lambda item: str(item.get("seed"))):
            lines.append(
                f"| `{mode}` | {row.get('seed')} | {fmt(row.get('baseline_top1'), 4)} | "
                f"{fmt(row.get('replacement_top1'), 4)} | {fmt(row.get('delta_top1'), 4)} | "
                f"{fmt(row.get('delta_loss'), 6)} | `{row.get('dir')}` |"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- Verdict: `{verdict}`.")
    if "address_lut" in summary and "shuffled_address_lut" in summary:
        addr = float(summary["address_lut"]["mean_delta_top1"])
        shuf = float(summary["shuffled_address_lut"]["mean_delta_top1"])
        lines.append(
            "- Address-alignment check: "
            f"`address_lut` mean Delta Acc@1 is {fmt(addr, 4)}, while `shuffled_address_lut` is {fmt(shuf, 4)}."
        )
    lines.append("- Claim boundary: this supports or weakens an address-specific adapter ablation only; it does not support energy, latency, ImageNet, or full-wrapper claims.")
    lines.append("")
    lines.append("## Local Evidence Check")
    lines.append("")
    lines.append(f"- Report root: `{root}`.")
    lines.append("- Local artifact set should contain no checkpoint weights; verify with `find results -name '*.pth*' -o -name '*.pt' -o -name '*.ckpt'` before committing artifacts.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize QK-LUTFormer local results.")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--e3-count", type=int, default=12, help="Latest T=1 E3 runs to summarize")
    parser.add_argument("--all-checkpoints", action="store_true", help="Summarize T=1 E3 runs across all checkpoints")
    parser.add_argument("--group-alpha", action="store_true", help="Group E3 rows by mode and alpha")
    parser.add_argument(
        "--group-address-scale",
        action="store_true",
        help="Group E3 rows by mode and address residual scale",
    )
    parser.add_argument("--brief", action="store_true", help="Print compact summary only")
    parser.add_argument("--write-md", type=Path, help="Write a markdown report to this path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    results = root / "results"
    print(f"[qk-lut-analyze] root={root}")

    train_dir = latest_t1_train(results)
    if train_dir:
        train_best = best_summary_row(train_dir)
        epoch, best_top1, last_top1 = train_best
        print(f"[qk-lut-analyze] latest_t1_train={train_dir.name}")
        print(f"[qk-lut-analyze] train_best_epoch={epoch} train_best_top1={best_top1} train_last_top1={last_top1}")
    else:
        train_best = None
        print("[qk-lut-analyze] latest_t1_train=missing")

    e0_dir = latest_t1_e0(results)
    if e0_dir:
        e0 = load_json(e0_dir / "metrics.json")
        e0_family = e0.get("model", {}).get("family", "unknown")
        print(f"[qk-lut-analyze] latest_t1_e0={e0_dir.name} family={e0_family}")
        print(
            "[qk-lut-analyze] e0 coverage="
            f"{fmt(e0.get('address_coverage'), 6)} singleton={fmt(e0.get('singleton_fraction'), 6)} "
            f"conditional_variance={fmt(e0.get('conditional_variance'), 6)}"
        )
    else:
        e0 = None
        print("[qk-lut-analyze] latest_t1_e0=missing")

    e1_dir = latest_t1_e1(results)
    if e1_dir:
        e1 = load_json(e1_dir / "metrics.json")
        e1_family = e1.get("model", {}).get("family", "unknown")
        recon = e1.get("overall_reconstruction", {})
        print(f"[qk-lut-analyze] latest_t1_e1={e1_dir.name} family={e1_family}")
        print(
            "[qk-lut-analyze] e1 global_mse="
            f"{fmt(recon.get('global_mean_mse'), 6)} "
            f"address_mse={fmt(recon.get('address_lut_mse'), 6)} "
            f"relative_reduction_pct={fmt(100.0 * float(recon.get('address_relative_mse_reduction', 0.0)), 4)}"
        )
    else:
        print("[qk-lut-analyze] latest_t1_e1=missing")

    grouped: dict[str, list[dict[str, Any]]] = {}
    checkpoint_marker = None if args.all_checkpoints or train_dir is None else train_dir.name
    if checkpoint_marker:
        print(f"[qk-lut-analyze] e3_checkpoint_filter={checkpoint_marker}")
    else:
        print("[qk-lut-analyze] e3_checkpoint_filter=all")
    for run_dir in reversed(latest_t1_e3(results, args.e3_count, checkpoint_marker)):
        metrics = load_json(run_dir / "metrics.json")
        mode = metrics.get("adapter_summary", {}).get("mode") or metrics.get("adapter_config", {}).get("mode", "unknown")
        alpha = metrics.get("env_overrides", {}).get("QKFORMER_LUT_E3_ALPHA_INIT")
        if alpha is None:
            alpha = metrics.get("adapter_config", {}).get("alpha_init")
        address_scale = metrics.get("env_overrides", {}).get("QKFORMER_LUT_E3_ADDRESS_SCALE")
        if address_scale is None:
            address_scale = metrics.get("adapter_config", {}).get("address_scale")
        group_mode = mode
        if args.group_alpha:
            group_mode += f"@alpha={fmt(alpha, 3)}"
        if args.group_address_scale:
            group_mode += f"@address_scale={fmt(address_scale, 3)}"
        cls = metrics.get("classification", {})
        replacement = cls.get("replacement", {})
        seed = (
            metrics.get("env_overrides", {}).get("QKFORMER_LUT_E3_SEED")
            or metrics.get("adapter_config", {}).get("mode_seed")
            or metrics.get("experiment", {}).get("seed")
        )
        row = {
            "dir": run_dir.name,
            "seed": seed,
            "mode": mode,
            "alpha": alpha,
            "baseline_top1": cls.get("baseline", {}).get("top1"),
            "replacement_top1": replacement.get("top1"),
            "delta_top1": cls.get("delta", {}).get("top1"),
            "baseline_loss": cls.get("baseline", {}).get("loss"),
            "replacement_loss": replacement.get("loss"),
            "delta_loss": cls.get("delta", {}).get("loss"),
            "kl": replacement.get("kl_to_baseline"),
            "logit_mse": replacement.get("logit_mse"),
            "local_mse": replacement.get("local_mse"),
        }
        grouped.setdefault(group_mode, []).append(row)
        if not args.brief:
            print(
                "[qk-lut-analyze] e3 "
                f"dir={row['dir']} mode={mode} seed={row['seed']} "
                f"alpha={fmt(alpha, 3)} "
                f"top1={fmt(row['baseline_top1'], 2)}->{fmt(row['replacement_top1'], 2)} "
                f"delta={fmt(row['delta_top1'], 4)} loss_delta={fmt(row['delta_loss'], 6)} "
                f"kl={fmt(row['kl'], 6)} logit_mse={fmt(row['logit_mse'], 6)} local_mse={fmt(row['local_mse'], 6)}"
            )

    summary = summarize_rows(grouped)
    for mode, item in sorted(summary.items()):
            print(
                "[qk-lut-analyze] summary "
                f"mode={mode} n={item['n']} mean_delta_top1={fmt(item.get('mean_delta_top1'), 4)} "
                f"min_delta_top1={fmt(item.get('min_delta_top1'), 4)} max_delta_top1={fmt(item.get('max_delta_top1'), 4)} "
                f"mean_delta_loss={fmt(item.get('mean_delta_loss'), 6)}"
            )

    verdict = decide_verdict(summary)
    print(f"[qk-lut-analyze] verdict={verdict}")
    if args.write_md:
        report_path = args.write_md if args.write_md.is_absolute() else root / args.write_md
        write_markdown_report(report_path, root, train_dir, train_best, e0_dir, e0, grouped, summary, verdict)
        print(f"[qk-lut-analyze] wrote_report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
