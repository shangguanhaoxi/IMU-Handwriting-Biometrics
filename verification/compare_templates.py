#!/usr/bin/env python3
"""Evaluate multi-template verification using an EXISTING frozen OMC checkpoint.

Model weights, feature normalization and model selection remain unchanged.
Each scoring method gets its own threshold calibrated exclusively on dev writers.
Test writers are embedded/evaluated only after thresholds are finalized.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from omc.data import load_split_config
from omc.engine import get_device
from omc.gravity import load_gravity_config, validate_gravity_subjects
from omc.model import OMC
from verification.protocol import (
    SCORING_METHODS, choose_far_threshold, create_trials, load_protocol, score_trials,
)
from verification.run import embed_rows, grouped_scores, load_feature_records


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No trials to save: {path}")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_scores(cohort, embeddings, owners, protocol, method, top_k, threshold):
    trials = create_trials(
        cohort, embeddings, owners, protocol["enrollment_characters"],
        method=method, top_k=top_k,
    )
    metrics = score_trials(trials, threshold)
    held = set(protocol["unseen_test_characters"])
    metrics_held = score_trials(
        [r for r in trials if r["character"] in held], threshold
    )
    for row in trials:
        row["threshold"] = float(threshold)
        row["accepted"] = int(row["similarity"] >= threshold)
        row["decision_correct"] = int(row["accepted"] == row["genuine"])
    return trials, metrics, metrics_held


def compare_frozen(checkpoint, rows, protocol, device, batch_size):
    """Run three scoring methods on EXACTLY the same frozen encoder/embeddings."""
    model = OMC(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    stats = checkpoint["stats"]
    embeddings = {}
    metadata = {}
    # Build dev and test embeddings separately; never adapt normalization.
    for group in ("dev", "test"):
        owners = protocol[group + "_subjects"]
        cohort = [row for row in rows if row["subject"] in owners]
        metadata[group] = cohort
        embeddings[group] = embed_rows(model, cohort, stats, device, batch_size)

    methods = protocol.get("scoring_methods", list(SCORING_METHODS))
    if set(methods) != set(SCORING_METHODS) or len(methods) != len(SCORING_METHODS):
        raise ValueError("Protocol must compare all three scoring methods exactly once")
    top_k = int(protocol.get("top_k", 3))
    if top_k < 1 or top_k > len(protocol["enrollment_characters"]):
        raise ValueError("Invalid top_k")
    result = {}
    all_test_trials = []
    for method in methods:
        dev_trials = create_trials(
            metadata["dev"], embeddings["dev"], protocol["dev_subjects"],
            protocol["enrollment_characters"], method=method, top_k=top_k,
        )
        threshold = choose_far_threshold(dev_trials, protocol["target_dev_far"])
        dev = score_trials(dev_trials, threshold)
        test_trials, test, held = evaluate_scores(
            metadata["test"], embeddings["test"], protocol["test_subjects"],
            protocol, method, top_k, threshold,
        )
        all_test_trials.extend(test_trials)
        result[method] = {
            "threshold_from_dev": threshold,
            "dev": dev,
            "test_all_non_enrollment_characters": test,
            "test_unseen_characters_only": held,
            "test_per_owner": grouped_scores(
                test_trials, threshold, protocol["test_subjects"]
            ),
        }
    # Every method must evaluate the same genuine/impostor pairs.
    identities = {}
    for method in methods:
        identities[method] = {
            (r["owner"], r["query_subject"], r["character"], r["path"])
            for r in all_test_trials if r["method"] == method
        }
    if len({frozenset(v) for v in identities.values()}) != 1:
        raise AssertionError("Trial alignment differs across scoring methods")
    return result, all_test_trials


def main():
    parser = argparse.ArgumentParser(
        description="K=10 multi-template verification comparison using a frozen OMC"
    )
    parser.add_argument(
        "--checkpoint", default="runs/verification_5train_2dev_2test/best.pt"
    )
    parser.add_argument("--protocol", default="verification/multitemplate_protocol.json")
    parser.add_argument("--data_path", default="../excel")
    parser.add_argument("--split_config", default="imu_split_config.json")
    parser.add_argument("--gravity_config", default="imu_gravity_config.json")
    parser.add_argument("--output_dir", default="runs/verification_multitemplate_k10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = ("model_state", "architecture", "stats", "data_settings",
                "train_subjects", "dev_subjects", "test_subjects")
    if any(key not in ck for key in required):
        raise ValueError("Expected an unseen-writer verification checkpoint (not closed-set)")
    char_cfg = load_split_config(args.split_config)
    active = char_cfg["active_subjects"]
    protocol = load_protocol(args.protocol, active, char_cfg)
    for group in ("train", "dev", "test"):
        if protocol[group + "_subjects"] != ck[group + "_subjects"]:
            raise ValueError(f"Writer partition differs from frozen checkpoint: {group}")
    if protocol.get("enrollment_count", 10) != 10:
        raise ValueError("This experiment requires enrollment_count=10")
    if len(protocol["enrollment_characters"]) != 10:
        raise ValueError("This experiment requires exactly 10 enrollment characters")
    if protocol.get("top_k", 3) != 3:
        raise ValueError("This experiment requires top_k=3")
    gravity = load_gravity_config(args.gravity_config)
    validate_gravity_subjects(gravity, active)
    settings = ck["data_settings"]
    if settings["gravity_config_digest"] != gravity["_digest"]:
        raise ValueError("Gravity configuration differs from checkpoint")
    # Do not refit model weights, data normalization, VAT thresholds, or model epoch.
    settings_args = argparse.Namespace(
        data_path=args.data_path,
        sample_rate=settings["sample_rate"],
        min_length=settings["min_length"],
        input_mode=settings["input_mode"],
        seq_len=settings["seq_len"],
        velocity_detrend=settings["velocity_detrend"],
        allow_skipped=False,
    )
    rows, ignored, skipped = load_feature_records(
        settings_args, active, char_cfg, gravity
    )
    if skipped:
        raise ValueError("Cannot compare methods with missing data")
    device = get_device(args.device)
    results, all_trials = compare_frozen(
        ck, rows, protocol, device, args.batch_size
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "test_trials_by_method.csv", all_trials)
    summary_rows = []
    for method, result in results.items():
        for subset in ("dev", "test_all_non_enrollment_characters",
                       "test_unseen_characters_only"):
            met = result[subset]
            summary_rows.append({
                "method": method,
                "subset": subset,
                "threshold_from_dev": result["threshold_from_dev"],
                "enrollment_count": len(protocol["enrollment_characters"]),
                "top_k": protocol["top_k"] if method == "topk_mean" else "",
                **met,
            })
    write_csv(output / "comparison.csv", summary_rows)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": ck.get("epoch"),
        "protocol": protocol,
        "number_of_feature_records": len(rows),
        "skipped": skipped,
        "ignored": ignored,
        "methods": results,
        "limitations": [
            "Same pre-existing encoder: isolates enrollment scoring from training effects.",
            "All three thresholds are independently calibrated on development writers only.",
            "The development and final test groups each contain only two real people.",
            "Repeated query and impostor trials are not statistically independent.",
            "Previously inspected test writers make this an exploratory comparison, not a fresh blind evaluation.",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Saved comparison to: {output.resolve()}")


if __name__ == "__main__":
    main()
