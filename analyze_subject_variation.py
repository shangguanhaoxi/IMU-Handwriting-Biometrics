#!/usr/bin/env python3
"""Diagnose subject-specific IMU motion variation and cross-character stability.

This is an exploratory analysis tool. It does not train a model or modify the
training cache. Raw channel units are preserved; no unit conversion is assumed.
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from omc.data import DirectoryReader, _integrate, build_features, discover_records, read_record
from omc.gravity import correct_gravity, load_gravity_config, validate_gravity_subjects

LOG = logging.getLogger("subject_analysis")
AXES = ("x", "y", "z")
BLOCKS = ("acc", "gyro")
SEQ_CHANNELS = (
    "acc_x", "acc_y", "acc_z",
    "velocity_x", "velocity_y", "velocity_z",
    "jerk_x", "jerk_y", "jerk_z",
    "acc_norm", "velocity_norm", "jerk_norm",
    "gyro_x", "gyro_y", "gyro_z",
    "angular_acc_x", "angular_acc_y", "angular_acc_z",
    "gyro_norm", "angular_acc_norm",
)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _feature_names() -> list[str]:
    names: list[str] = []
    bands = ((0, 2), (2, 5), (5, 15), (15, 50))
    for block in BLOCKS:
        for axis in AXES:
            names.append(f"emf_{block}_{axis}_spectral_centroid_hz")
            for lo, hi in bands:
                names.append(f"emf_{block}_{axis}_band_{lo}_{hi}hz_ratio")
        for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
            names.append(f"emf_{block}_cov_{AXES[i]}_{AXES[j]}")
        for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
            names.append(f"emf_{block}_spearman_{a}_{b}")
    for a in AXES:
        for g in AXES:
            names.append(f"emf_acc_gyro_cov_{a}_{g}")
    names.append("emf_log_duration")
    if len(names) != 58:
        raise RuntimeError(f"Expected 58 EMF global feature names, got {len(names)}")
    return names


EMF_NAMES = _feature_names()


def _record_features(x: np.ndarray, t: np.ndarray, seq_len: int, input_mode: str) -> dict:
    """Summarize raw six-axis signals and the same EMF features used by OMC."""
    if x.ndim != 2 or x.shape[1] < 6:
        raise ValueError(f"Expected six IMU channels, got shape {x.shape}")
    if len(x) < 4 or len(t) != len(x):
        raise ValueError(f"Insufficient samples or timestamp mismatch: x={x.shape}, t={t.shape}")

    acc = x[:, :3].astype(np.float64)
    gyro = x[:, 3:6].astype(np.float64)
    duration = float(t[-1] - t[0])
    dt = np.diff(t)
    if duration <= 0 or len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("Non-positive duration or non-monotonic timestamps")

    result: dict[str, float | int] = {
        "n_samples": int(len(x)),
        "duration_s": duration,
        "effective_sample_rate_hz": float((len(x) - 1) / duration),
        "median_dt_s": float(np.median(dt)),
        "dt_cv": float(np.std(dt) / max(np.mean(dt), 1e-12)),
    }

    for block_name, block in (("acc", acc), ("gyro", gyro)):
        for j, axis in enumerate(AXES):
            signal = block[:, j]
            result[f"{block_name}_{axis}_mean"] = float(np.mean(signal))
            result[f"{block_name}_{axis}_std"] = float(np.std(signal))
            result[f"{block_name}_{axis}_rms"] = float(np.sqrt(np.mean(signal ** 2)))
            result[f"{block_name}_{axis}_p05"] = float(np.quantile(signal, 0.05))
            result[f"{block_name}_{axis}_p95"] = float(np.quantile(signal, 0.95))
        magnitude = np.linalg.norm(block, axis=1)
        result[f"{block_name}_norm_mean"] = float(np.mean(magnitude))
        result[f"{block_name}_norm_std"] = float(np.std(magnitude))
        result[f"{block_name}_norm_rms"] = float(np.sqrt(np.mean(magnitude ** 2)))
        result[f"{block_name}_norm_p95"] = float(np.quantile(magnitude, 0.95))

    # Pearson correlations between each linear-acceleration and gyro axis.
    for i, acc_axis in enumerate(AXES):
        for j, gyro_axis in enumerate(AXES):
            result[f"corr_acc_{acc_axis}_gyro_{gyro_axis}"] = _safe_corr(acc[:, i], gyro[:, j])

    # Keep the same no-detrending integration convention as build_features.
    velocity = _integrate(acc, t, detrend=False)
    speed = np.linalg.norm(velocity, axis=1)
    result["velocity_proxy_mean"] = float(np.mean(speed))
    result["velocity_proxy_std"] = float(np.std(speed))
    result["velocity_proxy_rms"] = float(np.sqrt(np.mean(speed ** 2)))
    result["velocity_proxy_p95"] = float(np.quantile(speed, 0.95))

    seq, global_features, speed_from_pipeline = build_features(
        x, t, input_mode=input_mode, seq_len=seq_len, velocity_detrend=False
    )
    if len(global_features) != len(EMF_NAMES):
        raise ValueError(f"Expected 58 EMF features, got {len(global_features)}")
    for name, value in zip(EMF_NAMES, global_features):
        result[name] = float(value)

    # Summaries of all 20 EMF sequence channels. These are descriptive summaries,
    # not learned OMC embeddings.
    if seq.shape[1] != len(SEQ_CHANNELS):
        raise ValueError(f"Expected 20 EMF sequence channels, got {seq.shape}")
    for j, name in enumerate(SEQ_CHANNELS):
        channel = seq[:, j].astype(np.float64)
        result[f"seq_{name}_mean"] = float(np.mean(channel))
        result[f"seq_{name}_std"] = float(np.std(channel))
        result[f"seq_{name}_rms"] = float(np.sqrt(np.mean(channel ** 2)))
    result["velocity_proxy_pipeline"] = float(speed_from_pipeline)
    return result


def _aggregate_subject_split(records_df: pd.DataFrame) -> pd.DataFrame:
    excluded = {"subject", "character", "split", "path", "format"}
    numeric = [c for c in records_df.select_dtypes(include=[np.number]).columns if c not in excluded]
    grouped = records_df.groupby(["subject", "split"], dropna=False)[numeric].agg(["mean", "std", "count"])
    grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
    return grouped.reset_index()


def _pair_similarity(train_df: pd.DataFrame, feature_cols: list[str]):
    """Compare within-person cross-character similarity against between-person pairs."""
    if len(train_df) < 3:
        return pd.DataFrame(), pd.DataFrame()

    # Average repeated trials within each subject-character pair first, so a
    # character with more repeated recordings does not dominate pair counts.
    collapsed = train_df.groupby(["subject", "character"], as_index=False)[feature_cols].mean()
    matrix = collapsed[feature_cols].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median()).fillna(0.0).to_numpy(dtype=np.float64)
    matrix = StandardScaler().fit_transform(matrix)
    sims = cosine_similarity(matrix)
    subjects = collapsed["subject"].astype(str).to_numpy()
    chars = collapsed["character"].astype(str).to_numpy()

    groups = {
        "within_subject_different_character": [],
        "between_subject_same_character": [],
        "between_subject_different_character": [],
    }
    per_subject: dict[str, dict[str, list[float]]] = {}
    for subject in sorted(set(subjects)):
        per_subject[subject] = {
            "within_cross_character": [],
            "between_same_character": [],
            "between_different_character": [],
        }

    for i, j in combinations(range(len(collapsed)), 2):
        same_subject = subjects[i] == subjects[j]
        same_char = chars[i] == chars[j]
        value = float(sims[i, j])
        if same_subject and not same_char:
            groups["within_subject_different_character"].append(value)
            per_subject[subjects[i]]["within_cross_character"].append(value)
        elif not same_subject and same_char:
            groups["between_subject_same_character"].append(value)
            per_subject[subjects[i]]["between_same_character"].append(value)
            per_subject[subjects[j]]["between_same_character"].append(value)
        elif not same_subject and not same_char:
            groups["between_subject_different_character"].append(value)
            per_subject[subjects[i]]["between_different_character"].append(value)
            per_subject[subjects[j]]["between_different_character"].append(value)

    summary_rows = []
    for group, values in groups.items():
        arr = np.asarray(values, dtype=np.float64)
        summary_rows.append({
            "pair_type": group,
            "n_pairs": int(len(arr)),
            "cosine_similarity_mean": float(np.mean(arr)) if len(arr) else np.nan,
            "cosine_similarity_median": float(np.median(arr)) if len(arr) else np.nan,
            "cosine_similarity_std": float(np.std(arr)) if len(arr) else np.nan,
        })

    subject_rows = []
    for subject, groups_for_subject in per_subject.items():
        row = {"subject": subject}
        for name, values in groups_for_subject.items():
            arr = np.asarray(values, dtype=np.float64)
            row[f"{name}_n_pairs"] = int(len(arr))
            row[f"{name}_cosine_mean"] = float(np.mean(arr)) if len(arr) else np.nan
            row[f"{name}_cosine_std"] = float(np.std(arr)) if len(arr) else np.nan
        subject_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(subject_rows)


def _focus_group_comparison(records_df: pd.DataFrame, focus_subjects: list[str], out: Path):
    """Equal-weight subject-level descriptive comparison on the training split."""
    train = records_df[records_df["split"] == "train"].copy()
    numeric = [c for c in train.select_dtypes(include=[np.number]).columns]
    if train.empty or not numeric:
        return
    per_subject = train.groupby("subject")[numeric].mean()
    focus = [s for s in focus_subjects if s in per_subject.index]
    other = [s for s in per_subject.index if s not in focus]
    rows = []
    for feature in numeric:
        a = per_subject.loc[focus, feature].dropna() if focus else pd.Series(dtype=float)
        b = per_subject.loc[other, feature].dropna() if other else pd.Series(dtype=float)
        rows.append({
            "feature": feature,
            "focus_subjects": ";".join(focus),
            "focus_n_subjects": int(a.size),
            "focus_mean_across_subjects": float(a.mean()) if len(a) else np.nan,
            "focus_std_across_subjects": float(a.std(ddof=0)) if len(a) else np.nan,
            "other_n_subjects": int(b.size),
            "other_mean_across_subjects": float(b.mean()) if len(b) else np.nan,
            "other_std_across_subjects": float(b.std(ddof=0)) if len(b) else np.nan,
            "focus_minus_other_mean": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
        })
    pd.DataFrame(rows).to_csv(out / "focus_group_vs_others_train.csv", index=False, encoding="utf-8-sig")


def _prediction_analysis(prediction_path: Path, records_df: pd.DataFrame, subjects: list[str], out: Path):
    pred = pd.read_csv(prediction_path, encoding="utf-8-sig")
    required = {"path", "true_subject", "predicted_subject"}
    if not required.issubset(pred.columns):
        raise ValueError(
            f"Predictions file must contain columns {sorted(required)}; found {list(pred.columns)}. "
            "Use predictions.csv generated by test.py."
        )
    merged = pred.merge(records_df, on="path", how="left", suffixes=("", "_raw"), validate="one_to_one")
    missing = int(merged["duration_s"].isna().sum())
    if missing:
        LOG.warning("%d prediction rows could not be matched to parsed records", missing)
    merged["correct"] = merged["true_subject"].astype(str) == merged["predicted_subject"].astype(str)
    errors = merged.loc[~merged["correct"]].copy()
    errors.to_csv(out / "error_samples.csv", index=False, encoding="utf-8-sig")

    labels = [s for s in subjects if s in set(merged["true_subject"].astype(str))]
    y_true = merged["true_subject"].astype(str)
    y_pred = merged["predicted_subject"].astype(str)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    class_rows = []
    for i, label in enumerate(labels):
        class_rows.append({
            "subject": label,
            "support": int(support[i]),
            "correct": int(((y_true == label) & (y_pred == label)).sum()),
            "accuracy_recall": float(recall[i]),
            "precision": float(precision[i]),
            "f1": float(f1[i]),
        })
    pd.DataFrame(class_rows).to_csv(out / "per_subject_prediction_metrics.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "prediction_confusion_matrix_labeled.csv", encoding="utf-8-sig")
    pair_counts = errors.groupby(["true_subject", "predicted_subject"]).size().reset_index(name="n_errors")
    pair_counts = pair_counts.sort_values("n_errors", ascending=False)
    pair_counts.to_csv(out / "confusion_pairs.csv", index=False, encoding="utf-8-sig")

    # The current OMC checkpoint may also emit VAT speed-class predictions.
    if {"velocity_label", "velocity_prediction"}.issubset(merged.columns):
        vy = pd.to_numeric(merged["velocity_label"], errors="coerce")
        vp = pd.to_numeric(merged["velocity_prediction"], errors="coerce")
        valid = vy.notna() & vp.notna()
        if valid.any():
            vy = vy[valid].astype(int)
            vp = vp[valid].astype(int)
            velocity_classes = sorted(set(vy.tolist()) | set(vp.tolist()))
            vat_cm = confusion_matrix(vy, vp, labels=velocity_classes)
            vat_precision, vat_recall, vat_f1, vat_support = precision_recall_fscore_support(
                vy, vp, labels=velocity_classes, zero_division=0
            )
            vat_report = {
                "n_samples": int(len(vy)),
                "accuracy": float(accuracy_score(vy, vp)),
                "macro_f1": float(f1_score(vy, vp, labels=velocity_classes, average="macro", zero_division=0)),
                "classes": velocity_classes,
                "class_support": [int(x) for x in vat_support],
            }
            (out / "vat_metrics.json").write_text(
                json.dumps(vat_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            pd.DataFrame({
                "velocity_class": velocity_classes,
                "support": vat_support.astype(int),
                "precision": vat_precision,
                "recall": vat_recall,
                "f1": vat_f1,
            }).to_csv(out / "vat_per_class_metrics.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(vat_cm, index=velocity_classes, columns=velocity_classes).to_csv(
                out / "vat_confusion_matrix.csv", encoding="utf-8-sig"
            )
            vat_rows = []
            valid_rows = merged.loc[valid].copy()
            for subject, group in valid_rows.groupby("true_subject"):
                vat_rows.append({
                    "subject": subject,
                    "n_samples": int(len(group)),
                    "vat_accuracy": float(
                        (pd.to_numeric(group["velocity_label"]) ==
                         pd.to_numeric(group["velocity_prediction"])).mean()
                    ),
                })
            pd.DataFrame(vat_rows).to_csv(
                out / "vat_per_subject_metrics.csv", index=False, encoding="utf-8-sig"
            )

    numeric = [c for c in records_df.select_dtypes(include=[np.number]).columns]
    if numeric:
        error_summary = merged.groupby("correct")[numeric].agg(["mean", "std", "count"])
        error_summary.columns = [f"{name}_{stat}" for name, stat in error_summary.columns]
        error_summary.reset_index().to_csv(out / "test_correct_vs_error_feature_summary.csv", index=False, encoding="utf-8-sig")
    return merged, pd.DataFrame(class_rows), cm, labels


def _make_plots(records_df: pd.DataFrame, similarity_df: pd.DataFrame,
                prediction_df: pd.DataFrame | None, cm: np.ndarray | None,
                labels: list[str] | None, out: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib is unavailable; CSV/JSON outputs were written but plots were skipped")
        return

    train = records_df[records_df["split"] == "train"]
    if not train.empty:
        summary = train.groupby("subject")[["acc_norm_rms", "gyro_norm_rms"]].mean().sort_index()
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        summary["acc_norm_rms"].plot(kind="bar", ax=axes[0])
        axes[0].set_title("Training-set acceleration magnitude RMS by subject")
        axes[0].set_ylabel("Acceleration RMS (source unit)")
        summary["gyro_norm_rms"].plot(kind="bar", ax=axes[1])
        axes[1].set_title("Training-set gyroscope magnitude RMS by subject")
        axes[1].set_ylabel("Gyroscope RMS (source unit)")
        axes[1].set_xlabel("Subject")
        fig.tight_layout()
        fig.savefig(out / "subject_motion_rms.png", dpi=160)
        plt.close(fig)

        scatter = train.groupby("subject")[["velocity_proxy_mean", "duration_s"]].mean()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(scatter["duration_s"], scatter["velocity_proxy_mean"])
        for subject, row in scatter.iterrows():
            ax.annotate(str(subject), (row["duration_s"], row["velocity_proxy_mean"]))
        ax.set_xlabel("Mean writing duration (s)")
        ax.set_ylabel("Mean integrated-acceleration speed proxy (source unit·s)")
        ax.set_title("Training-set speed proxy versus writing duration")
        fig.tight_layout()
        fig.savefig(out / "subject_velocity_duration.png", dpi=160)
        plt.close(fig)

    if not similarity_df.empty:
        plot_df = similarity_df[similarity_df["n_pairs"] > 0]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(plot_df["pair_type"], plot_df["cosine_similarity_mean"])
        ax.set_ylabel("Mean cosine similarity (standardized EMF summaries)")
        ax.set_title("Within-identity and between-identity feature similarity")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(out / "similarity_pair_groups.png", dpi=160)
        plt.close(fig)

    if prediction_df is not None and cm is not None and labels is not None:
        per_subject = prediction_df.groupby("true_subject")["correct"].mean().reindex(labels)
        fig, ax = plt.subplots(figsize=(11, 5))
        per_subject.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Test accuracy")
        ax.set_title("Test accuracy by true subject")
        fig.tight_layout()
        fig.savefig(out / "test_accuracy_by_subject.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(cm, interpolation="nearest", aspect="auto")
        fig.colorbar(image, ax=ax, label="Number of samples")
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        ax.set_xlabel("Predicted subject")
        ax.set_ylabel("True subject")
        ax.set_title("Test confusion matrix")
        fig.tight_layout()
        fig.savefig(out / "test_confusion_matrix.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze subject-specific IMU motion variation and cross-character stability.")
    parser.add_argument("--data_path", default="../excel", help="Extracted IMU data directory")
    parser.add_argument("--split_config", default="imu_split_config.json")
    parser.add_argument("--gravity_config", default="imu_gravity_config.json")
    parser.add_argument("--output_dir", default="analysis/subject_diagnostics")
    parser.add_argument("--predictions", default="", help="Optional predictions.csv from test.py for error analysis")
    parser.add_argument("--input_mode", choices=["accgyro6"], default="accgyro6", help="This diagnostic currently targets the six-axis experiment")
    parser.add_argument("--focus_subjects", default="A", help="Comma-separated active writers to compare descriptively against the others; defaults to A")
    parser.add_argument("--sample_rate", type=float, default=100.0)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--min_length", type=int, default=16)
    parser.add_argument("--split_for_similarity", default="train", choices=["train", "val", "test"])
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    data_root = Path(args.data_path).resolve()
    split_path = Path(args.split_config).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    records, ignored, subjects = discover_records(data_root, split_path)
    gravity_cfg = load_gravity_config(args.gravity_config)
    validate_gravity_subjects(gravity_cfg, subjects)
    reader = DirectoryReader(data_root)
    rows = []
    skipped = []
    for index, record in enumerate(records, start=1):
        try:
            x, t = read_record(reader, record["archive_path"], args.sample_rate)
            if len(x) < args.min_length:
                raise ValueError(f"too_short: {len(x)} < {args.min_length}")
            raw_mean_vector = float(np.linalg.norm(np.mean(x[:, :3], axis=0)))
            x, estimated_gravity = correct_gravity(x, t, record["subject"], gravity_cfg)
            features = _record_features(x, t, args.seq_len, args.input_mode)
            features["raw_acc_mean_vector_norm"] = raw_mean_vector
            features["estimated_gravity_mean_vector_norm"] = float(
                np.linalg.norm(np.mean(estimated_gravity, axis=0))
            )
            features["corrected_acc_mean_vector_norm"] = float(
                np.linalg.norm(np.mean(x[:, :3], axis=0))
            )
            rows.append({
                "path": record["archive_path"],
                "subject": record["subject"],
                "character": record["character"],
                "split": record["split"],
                "format": record["format"],
                **features,
            })
        except Exception as exc:
            skipped.append({"path": record["archive_path"], "reason": str(exc)})
        if index % 100 == 0:
            LOG.info("Parsed %d/%d discovered records", index, len(records))

    if not rows:
        raise RuntimeError("No valid recordings were parsed")
    records_df = pd.DataFrame(rows).sort_values(["subject", "split", "character", "path"]).reset_index(drop=True)
    records_df.to_csv(out / "record_level_features.csv", index=False, encoding="utf-8-sig")
    _aggregate_subject_split(records_df).to_csv(out / "subject_split_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped).to_csv(out / "analysis_skipped_records.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ignored).to_csv(out / "ignored_files.csv", index=False, encoding="utf-8-sig")

    feature_cols = [c for c in records_df.columns if c.startswith("emf_") or c.startswith("seq_")]
    focus_subjects = [s.strip() for s in args.focus_subjects.split(",") if s.strip()]
    _focus_group_comparison(records_df, focus_subjects, out)
    similarity_input = records_df[records_df["split"] == args.split_for_similarity].copy().reset_index(drop=True)
    similarity_df, subject_similarity_df = _pair_similarity(similarity_input, feature_cols)
    similarity_df.to_csv(out / "feature_similarity_pair_summary.csv", index=False, encoding="utf-8-sig")
    subject_similarity_df.to_csv(out / "per_subject_cross_character_similarity.csv", index=False, encoding="utf-8-sig")

    prediction_df = None
    cm = None
    labels = None
    if args.predictions:
        prediction_path = Path(args.predictions).resolve()
        prediction_df, _, cm, labels = _prediction_analysis(prediction_path, records_df, subjects, out)

    report = {
        "data_path": str(data_root),
        "records_discovered": len(records),
        "records_parsed": len(records_df),
        "records_skipped": len(skipped),
        "ignored_files": len(ignored),
        "subjects": subjects,
        "split_counts_parsed": {str(k): int(v) for k, v in records_df["split"].value_counts().to_dict().items()},
        "focus_subjects": focus_subjects,
        "gravity_config_digest": gravity_cfg["_digest"],
        "gravity_config": {k: v for k, v in gravity_cfg.items() if k != "_digest"},
        "split_for_similarity": args.split_for_similarity,
        "similarity_feature_count": len(feature_cols),
        "similarity_note": "Repeated trials are averaged within each subject-character pair. Cosine similarity is computed on standardized handcrafted EMF global features plus per-channel sequence mean/std/RMS; it is not an OMC learned embedding.",
        "focus_group_note": "The focus-versus-other comparison uses equal-weight subject-level means from the training split and is descriptive, not a significance test.",
        "velocity_note": "velocity_proxy is obtained by integrating gravity-removed linear acceleration with zero initial velocity and no drift correction; it is a proxy, not ground-truth pen velocity.",
        "unit_note": "Raw acceleration and gyroscope units are preserved from source spreadsheets; no unit conversion is assumed.",
        "correction_note": "EMF and velocity summaries are computed on corrected acceleration. The raw_acc_mean_vector_norm metric records the pre-correction acceleration offset; the correction is only a low-frequency estimate of gravity, not ground truth.",
        "skipped": skipped,
    }
    (out / "analysis_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _make_plots(records_df, similarity_df, prediction_df, cm, labels, out)

    print("\n========== SUBJECT ANALYSIS REPORT ==========")
    print(f"Discovered: {len(records)}; parsed: {len(records_df)}; skipped: {len(skipped)}; subjects: {len(subjects)}")
    print("Parsed split counts:", records_df["split"].value_counts().to_dict())
    print("Output directory:", out)
    print("Key files: record_level_features.csv, subject_split_summary.csv, feature_similarity_pair_summary.csv, per_subject_cross_character_similarity.csv, focus_group_vs_others_train.csv")
    if args.predictions:
        print("Prediction files: per_subject_prediction_metrics.csv, error_samples.csv, confusion_pairs.csv, prediction_confusion_matrix_labeled.csv")
    if skipped:
        print("Skipped records:")
        for item in skipped:
            print(item)


if __name__ == "__main__":
    main()
