#!/usr/bin/env python3
"""Pilot unseen-writer, cross-character IMU verification.

This is intentionally separate from the closed-set train.py/test.py.
Training writers, development writers, and test writers are disjoint.
The OMC encoder is frozen for enrollment and verification. No test labels,
scores or impostor trials are used for checkpoint/threshold selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from omc.data import (
    DirectoryReader, FeatureDataset, build_features, discover_records,
    fit_stats, load_split_config, read_record,
)
from omc.engine import fix_seed, get_device, train_epoch
from omc.gravity import correct_gravity, load_gravity_config, validate_gravity_subjects
from omc.model import BoundedUncertaintyWeighting, OMC
from verification.protocol import (
    choose_far_threshold, create_trials, load_protocol, score_trials,
)

LOG = logging.getLogger("verification")


def load_feature_records(args, active_subjects, character_cfg, gravity_cfg):
    records, ignored, discovered = discover_records(args.data_path, character_cfg)
    if set(discovered) != set(active_subjects):
        raise ValueError("Active writers differ from discovered writers")
    reader = DirectoryReader(args.data_path)
    rows, skipped = [], []
    for r in records:
        try:
            x, t = read_record(reader, r["archive_path"], args.sample_rate)
            if len(x) < args.min_length:
                raise ValueError(f"Too short: {len(x)} < {args.min_length}")
            x, _ = correct_gravity(x, t, r["subject"], gravity_cfg)
            seq, glob, speed = build_features(
                x, t, args.input_mode, args.seq_len, args.velocity_detrend
            )
            rows.append({
                "subject": r["subject"], "character": r["character"],
                "path": r["archive_path"], "split": r["split"],
                "seq": seq, "global": glob, "speed": speed,
            })
        except Exception as exc:
            LOG.warning("Skipping %s: %s", r["archive_path"], exc)
            skipped.append({"path": r["archive_path"], "error": str(exc)})
    if skipped and not args.allow_skipped:
        raise RuntimeError(
            f"{len(skipped)} records failed preprocessing. Fix source data, "
            "or rerun with --allow_skipped and inspect skipped_records.json."
        )
    return rows, ignored, skipped


def make_data(rows, protocol):
    train_writers = protocol["train_subjects"]
    train_chars = set(protocol.get("_train_characters", []))
    train = [r for r in rows if r["subject"] in train_writers and r["character"] in train_chars]
    if not train:
        raise ValueError("No model-training recordings")
    ids = {name: i for i, name in enumerate(train_writers)}
    stats = fit_stats(train)  # Critically: never fit on dev/test writers
    velocity_thresholds = np.quantile([r["speed"] for r in train], [1 / 3, 2 / 3]).tolist()
    for r in rows:
        r["identity"] = ids.get(r["subject"], 0)  # dev/test identity unused by encoder
        r["velocity"] = int(np.searchsorted(velocity_thresholds, r["speed"], side="right"))
    return train, stats, velocity_thresholds


class EncoderInputDataset(torch.utils.data.Dataset):
    """Label-free inference dataset for unseen writers.

    Uses checkpoint's frozen TRAIN-writer normalization; unlike FeatureDataset
    it does not require identity or auxiliary velocity labels.
    """
    def __init__(self, rows, stats):
        self.rows, self.stats = rows, stats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = self.rows[index]
        seq = (item["seq"] - self.stats["seq_mean"]) / self.stats["seq_std"]
        glob = (item["global"] - self.stats["global_mean"]) / self.stats["global_std"]
        return {
            "seq": torch.as_tensor(seq, dtype=torch.float32),
            "global": torch.as_tensor(glob, dtype=torch.float32),
        }


@torch.no_grad()
def embed_rows(model, rows, stats, device, batch_size):
    if not rows:
        raise ValueError("No rows to embed")
    model.eval()
    loader = DataLoader(EncoderInputDataset(rows, stats), batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in loader:
        z, _ = model(batch["seq"].to(device), batch["global"].to(device))
        embeddings.append(z.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def measure_group(model, rows, stats, device, protocol, group, batch_size):
    subjects = protocol[group + "_subjects"]
    cohort = [r for r in rows if r["subject"] in subjects]
    vectors = embed_rows(model, cohort, stats, device, batch_size)
    return create_trials(cohort, vectors, subjects, protocol["enrollment_characters"])


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No records for {path}")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def grouped_scores(trials, threshold, subjects):
    result = {}
    for owner in subjects:
        subset = [t for t in trials if t["owner"] == owner]
        result[owner] = score_trials(subset, threshold)
    return result


def main():
    p = argparse.ArgumentParser(description="Disjoint-writer cross-character IMU verification pilot")
    p.add_argument("--data_path", default="../excel")
    p.add_argument("--split_config", default="imu_split_config.json")
    p.add_argument("--gravity_config", default="imu_gravity_config.json")
    p.add_argument("--protocol", default="verification/protocol.json")
    p.add_argument("--output_dir", default="runs/verification_5train_2dev_2test")
    p.add_argument("--input_mode", choices=["accgyro6", "acc3"], default="accgyro6")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--sample_rate", type=float, default=100.)
    p.add_argument("--min_length", type=int, default=16)
    p.add_argument("--velocity_detrend", action="store_true")
    p.add_argument("--allow_skipped", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--ff_mult", type=int, default=2)
    p.add_argument("--conv_kernel", type=int, default=15)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--ort_lambda", type=float, default=.1)
    p.add_argument("--arc_scale", type=float, default=30.)
    p.add_argument("--arc_margin", type=float, default=.35)
    p.add_argument("--no_emf", action="store_true")
    p.add_argument("--no_vat", action="store_true")
    p.add_argument("--no_arcface", action="store_true")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fix_seed(a.seed)
    device = get_device(a.device)
    cfg = load_split_config(a.split_config)
    active = cfg["active_subjects"]
    protocol = load_protocol(a.protocol, active, cfg)
    protocol["_train_characters"] = cfg["train_characters"]
    gravity = load_gravity_config(a.gravity_config)
    validate_gravity_subjects(gravity, active)

    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, ignored, skipped = load_feature_records(a, active, cfg, gravity)
    (output / "skipped_records.json").write_text(
        json.dumps({"ignored": ignored, "skipped": skipped}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    train, stats, thresholds = make_data(rows, protocol)

    # Verify availability of the full enrollment set before training.
    for group in ("dev", "test"):
        for subject in protocol[group + "_subjects"]:
            present = {r["character"] for r in rows if r["subject"] == subject}
            if not set(protocol["enrollment_characters"]) <= present:
                raise ValueError(f"Insufficient enrollment records: {subject}")
            if not (present - set(protocol["enrollment_characters"])):
                raise ValueError(f"No held-out query characters: {subject}")

    first = train[0]
    arch = {
        "seq_dim": first["seq"].shape[-1],
        "global_dim": len(first["global"]),
        "num_classes": len(protocol["train_subjects"]),
        "dim": a.dim, "depth": a.depth, "heads": a.heads,
        "embed_dim": a.embed_dim, "ff_mult": a.ff_mult,
        "conv_kernel": a.conv_kernel, "dropout": a.dropout,
        "use_emf": not a.no_emf, "use_vat": not a.no_vat,
        "input_mode": a.input_mode,
    }
    model = OMC(**arch).to(device)
    model.arcface.scale, model.arcface.margin = a.arc_scale, a.arc_margin
    awl = BoundedUncertaintyWeighting().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(awl.parameters()),
        lr=a.lr, weight_decay=a.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(a.epochs, 1), eta_min=a.lr * .05
    )
    loader = DataLoader(
        FeatureDataset(train, stats), batch_size=a.batch_size,
        shuffle=True, pin_memory=device.type == "cuda",
    )
    print(
        f"device={device}; train_writers={protocol['train_subjects']}; "
        f"dev_writers={protocol['dev_subjects']}; test_writers={protocol['test_subjects']}; "
        f"train_records={len(train)}; all_processed={len(rows)}",
        flush=True,
    )
    best_auc, best_eer, best_epoch, stale = -1., np.inf, 0, 0
    history = []
    for epoch in range(1, a.epochs + 1):
        loss = train_epoch(model, awl, loader, optimizer, device, a.ort_lambda, not a.no_arcface)
        schedule.step()
        # Only development writers; the FINAL test group is untouched here.
        dev_trials = measure_group(model, rows, stats, device, protocol, "dev", a.batch_size)
        dev_threshold = choose_far_threshold(dev_trials, protocol["target_dev_far"])
        dev_metrics = score_trials(dev_trials, dev_threshold)
        history.append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "dev_auc": dev_metrics["roc_auc"], "dev_eer": dev_metrics["eer_approx"],
            "dev_far": dev_metrics["far"], "dev_frr": dev_metrics["frr"],
            **{"loss_" + k: v for k, v in loss.items()},
        })
        print(
            f"epoch={epoch:03d} train_loss={loss['total']:.4f} "
            f"dev_auc={dev_metrics['roc_auc']:.4f} "
            f"dev_eer={dev_metrics['eer_approx']:.4f} "
            f"dev_far={dev_metrics['far']:.4f} dev_frr={dev_metrics['frr']:.4f}",
            flush=True,
        )
        improved = dev_metrics["roc_auc"] > best_auc + 1e-6 or (
            abs(dev_metrics["roc_auc"] - best_auc) <= 1e-6
            and dev_metrics["eer_approx"] < best_eer - 1e-6
        )
        if improved:
            best_auc, best_eer, best_epoch, stale = (
                dev_metrics["roc_auc"], dev_metrics["eer_approx"], epoch, 0
            )
            checkpoint = {
                "model_state": model.state_dict(), "architecture": arch,
                "stats": stats, "train_subjects": protocol["train_subjects"],
                "dev_subjects": protocol["dev_subjects"], "test_subjects": protocol["test_subjects"],
                "protocol": {k: v for k, v in protocol.items() if not k.startswith("_")},
                "data_settings": {
                    "input_mode": a.input_mode, "seq_len": a.seq_len,
                    "sample_rate": a.sample_rate, "min_length": a.min_length,
                    "velocity_detrend": a.velocity_detrend,
                    "gravity_config_digest": gravity["_digest"],
                },
                "velocity_thresholds": thresholds,
                "epoch": epoch, "dev_metrics": dev_metrics,
                "dev_threshold": dev_threshold,
            }
            torch.save(checkpoint, output / "best.pt")
        else:
            stale += 1
        if stale >= a.patience:
            print(f"Early stopping; best_epoch={best_epoch}")
            break

    write_csv(output / "train_history.csv", history)
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    threshold = float(best["dev_threshold"])
    # FINAL test group is accessed only after checkpoint + threshold are frozen.
    test_trials = measure_group(model, rows, stats, device, protocol, "test", a.batch_size)
    test_metrics = score_trials(test_trials, threshold)
    heldout_characters = set(protocol["unseen_test_characters"])
    heldout_trials = [t for t in test_trials if t["character"] in heldout_characters]
    heldout_metrics = score_trials(heldout_trials, threshold)
    for trial in test_trials:
        trial["accepted"] = int(trial["similarity"] >= threshold)
        trial["decision_correct"] = int(trial["accepted"] == trial["genuine"])
    write_csv(output / "test_trials.csv", test_trials)
    report = {
        "protocol": {k: v for k, v in protocol.items() if not k.startswith("_")},
        "best_epoch": best_epoch,
        "dev_metrics_at_selection": best["dev_metrics"],
        "fixed_threshold_from_dev": threshold,
        "test_all_non_enrollment_characters": test_metrics,
        "test_unseen_characters_only": heldout_metrics,
        "test_per_owner": grouped_scores(test_trials, threshold, protocol["test_subjects"]),
        "record_counts": {
            "all_processed": len(rows), "training_records": len(train),
            "skipped": len(skipped),
        },
        "limits": [
            "Only 2 independent final test writers; verification trials are correlated.",
            "5% target FAR is exploratory, not a production unlock security guarantee.",
            "Test all-characters scores include query characters also seen for training writers.",
            "Test held-out character scores include only the 10 original test characters.",
            "No claim of security against spoofing/replay attacks.",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "run_config.json").write_text(
        json.dumps(vars(a), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
