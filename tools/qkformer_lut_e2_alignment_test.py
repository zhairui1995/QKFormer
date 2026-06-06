#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch

from qkformer_lut_e2_replace import (
    PrototypeBank,
    QKProjectionReplacer,
    build_cifar10_model,
    build_loader,
    calibrate_prototypes,
    evaluate_replacement,
    load_checkpoint_if_available,
    load_config,
    reset_model_state,
    set_seed,
)


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run(config_path: Path, output_dir: Path) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(config_path)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    model_cfg = dict(cfg["model"])
    diag_cfg = dict(cfg["diagnostic"])
    data_cfg = dict(cfg["data"])
    calibration_cfg = dict(data_cfg["calibration"])
    evaluation_cfg = dict(data_cfg["evaluation"])
    replacement_cfg = dict(cfg["replacement"])

    env_checkpoint = os.environ.get("QKFORMER_LUT_CKPT")
    if env_checkpoint:
        model_cfg["checkpoint"] = env_checkpoint
    env_time_step = os.environ.get("QKFORMER_LUT_TIME_STEP")
    if env_time_step:
        model_cfg["time_step"] = int(env_time_step)
    env_data_dir = os.environ.get("QKFORMER_LUT_DATA_DIR")
    if env_data_dir:
        calibration_cfg["data_dir"] = env_data_dir
        evaluation_cfg["data_dir"] = env_data_dir
    env_calib_batches = os.environ.get("QKFORMER_LUT_E2_CALIB_BATCHES")
    if env_calib_batches:
        calibration_cfg["num_batches"] = int(env_calib_batches)
    env_eval_batches = os.environ.get("QKFORMER_LUT_E2_EVAL_BATCHES")
    if env_eval_batches:
        evaluation_cfg["num_batches"] = int(env_eval_batches)
    env_targets = os.environ.get("QKFORMER_LUT_E2_TARGETS")
    if env_targets:
        replacement_cfg["target_modules"] = _split_csv(env_targets)
    env_blend = os.environ.get("QKFORMER_LUT_E2_BLEND")
    if env_blend:
        replacement_cfg["blend"] = float(env_blend)
    env_modes = os.environ.get("QKFORMER_LUT_E2_ALIGNMENT_MODES")
    modes = _split_csv(env_modes) if env_modes else ["address_lut", "shuffled_address_lut", "global_mean"]
    env_mode_seed = os.environ.get("QKFORMER_LUT_E2_MODE_SEED")
    mode_seed = int(env_mode_seed) if env_mode_seed else int(replacement_cfg.get("mode_seed", 0))

    device_name = str(diag_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the upstream cupy-backed QKFormer LIF modules")
    device = torch.device(device_name)
    model = build_cifar10_model(root, model_cfg)
    checkpoint_info = load_checkpoint_if_available(model, model_cfg.get("checkpoint"))
    model.to(device)
    model.eval()

    calibration_loader = build_loader(calibration_cfg, device)
    evaluation_loader = build_loader(evaluation_cfg, device)
    bank = PrototypeBank(min_count=int(diag_cfg.get("prototype_min_count", 2)))
    print("[qk-lut-e2-align] calibration_start")
    calibration_batches, calibration_hook_summary = calibrate_prototypes(
        model, calibration_loader, calibration_cfg, diag_cfg, bank, device
    )
    print(f"[qk-lut-e2-align] calibration_batches={calibration_batches}")

    target_modules = list(replacement_cfg.get("target_modules", [])) or ["stage1.0.tssa"]
    mode_results: Dict[str, object] = {}
    for mode in modes:
        print(f"[qk-lut-e2-align] eval_mode={mode}")
        reset_model_state(model)
        replacer = QKProjectionReplacer(
            model,
            bank.prototypes,
            target_modules,
            token_bins=int(diag_cfg.get("token_bins", 8)),
            channel_bins=int(diag_cfg.get("channel_bins", 8)),
            population_bins=int(diag_cfg.get("population_bins", 4)),
            blend=float(replacement_cfg.get("blend", 1.0)),
            mode=mode,
            mode_seed=mode_seed,
        )
        evaluation_batches, classification = evaluate_replacement(
            model, evaluation_loader, evaluation_cfg, replacer, device
        )
        local_replacement = replacer.summary()
        replacer.close()
        mode_results[mode] = {
            "classification": classification,
            "local_replacement": local_replacement,
            "evaluation_batches": evaluation_batches,
        }
        print(
            f"[qk-lut-e2-align] mode={mode} "
            f"delta_top1={classification['delta']['top1']} "
            f"delta_loss={classification['delta']['loss']} "
            f"kl={classification['replacement']['kl_to_baseline']} "
            f"logit_mse={classification['replacement']['logit_mse']}"
        )

    metrics = {
        "experiment": {"name": "qkformer_lut_e2_alignment_test", "seed": cfg["experiment"].get("seed", 42)},
        "model": {
            "family": model_cfg["family"],
            "time_step": model_cfg["time_step"],
            "layer": model_cfg["layer"],
            "dim": model_cfg["dim"],
            "num_heads": model_cfg["num_heads"],
            "checkpoint": checkpoint_info,
        },
        "data": {
            "calibration": {
                "source": "cifar10",
                "split": calibration_cfg.get("split"),
                "num_batches": calibration_batches,
                "batch_size": calibration_cfg["batch_size"],
                "shuffle": bool(calibration_cfg.get("shuffle", False)),
                "seed": calibration_cfg.get("seed"),
            },
            "evaluation": {
                "source": "cifar10",
                "split": evaluation_cfg.get("split"),
                "num_batches": next(iter(mode_results.values()))["evaluation_batches"] if mode_results else 0,
                "batch_size": evaluation_cfg["batch_size"],
            },
        },
        "diagnostic_config": diag_cfg,
        "replacement_config": {
            **replacement_cfg,
            "modes": modes,
            "mode_seed": mode_seed,
        },
        "env_overrides": {
            "QKFORMER_LUT_CKPT": env_checkpoint,
            "QKFORMER_LUT_TIME_STEP": env_time_step,
            "QKFORMER_LUT_E2_CALIB_BATCHES": env_calib_batches,
            "QKFORMER_LUT_E2_EVAL_BATCHES": env_eval_batches,
            "QKFORMER_LUT_E2_TARGETS": env_targets,
            "QKFORMER_LUT_E2_BLEND": env_blend,
            "QKFORMER_LUT_E2_ALIGNMENT_MODES": env_modes,
            "QKFORMER_LUT_E2_MODE_SEED": env_mode_seed,
        },
        "target_modules": target_modules,
        "mode_results": mode_results,
        "calibration_prototypes": bank.summary(),
        "calibration_hook_summary": calibration_hook_summary,
        "verdict": "PENDING_PHASE_GATE_REVIEW" if checkpoint_info["loaded"] else "PENDING_REAL_DATA_CHECKPOINT",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[qk-lut-e2-align] wrote {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QK-LUTFormer E2 frozen prototype address-alignment test")
    parser.add_argument("--config", type=Path, default=Path("configs/qkformer_lut_e2_replace.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
