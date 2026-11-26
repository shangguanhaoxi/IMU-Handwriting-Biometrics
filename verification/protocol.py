"""Utilities for unseen-writer IMU verification.

No model weights, threshold, or normalization parameters are fitted to final
test writers. A subject's enrollment and query characters never overlap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def load_protocol(path, active_subjects, split_config):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = [p["train_subjects"], p["dev_subjects"], p["test_subjects"]]
    if any(not isinstance(group, list) or len(group) < 2 for group in groups):
        raise ValueError("Each subject group needs at least two writers")
    flat = [s for group in groups for s in group]
    if len(flat) != len(set(flat)) or set(flat) != set(active_subjects):
        raise ValueError("Protocol must partition active subjects without duplicates")
    chars = set(split_config["train_characters"])
    enroll = p["enrollment_characters"]
    if not isinstance(enroll, list) or not enroll or len(enroll) != len(set(enroll)) or not set(enroll) <= chars:
        raise ValueError("Enrollment characters must be unique and belong to training characters")
    if p["query_policy"] != "all_characters_not_enrolled":
        raise ValueError("Unsupported query policy")
    held = p["unseen_test_characters"]
    if set(held) != set(split_config["test_characters"]):
        raise ValueError("The held-out character list must match split test_characters")
    far = float(p["target_dev_far"])
    if not 0 <= far < 1:
        raise ValueError("target_dev_far must be in [0,1)")
    return p


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("Zero-length embedding")
    return x / norms


SCORING_METHODS = ("mean_prototype", "max_similarity", "topk_mean")


def score_enrollment(query, enrollment_vectors, method="mean_prototype", top_k=3):
    """Cosine scores against one writer's independent normalized templates."""
    enrollment_vectors = normalize_rows(enrollment_vectors)
    query = normalize_rows(query)
    if enrollment_vectors.ndim != 2 or query.ndim != 1:
        raise ValueError("Expected a single query vector and a matrix of enrollment vectors")
    if enrollment_vectors.shape[1] != query.shape[0]:
        raise ValueError("Embedding dimension mismatch")
    if method not in SCORING_METHODS:
        raise ValueError(f"Unknown scoring method: {method}")
    similarities = enrollment_vectors @ query
    if method == "mean_prototype":
        prototype = normalize_rows(enrollment_vectors.mean(axis=0))
        return float(query @ prototype)
    if method == "max_similarity":
        return float(np.max(similarities))
    if not isinstance(top_k, int) or not 1 <= top_k <= len(similarities):
        raise ValueError("top_k must be in [1, enrollment_count]")
    return float(np.mean(np.sort(similarities)[-top_k:]))


def create_trials(rows, embeddings, owners, enrollment_characters, query_characters=None,
                  method="mean_prototype", top_k=3):
    """Construct writer-disjoint verification trials using independent templates.

    Enrollment characters are excluded from queries for EVERY query writer,
    including impostors. Every owner has the same requested enrollment character
    set; multiple trials per character are rejected to prevent a hidden change
    to K. Normalization uses the frozen model's embeddings only.
    """
    embeddings = normalize_rows(embeddings)
    if len(rows) != len(embeddings):
        raise ValueError("Rows and embeddings differ in length")
    if len(set(owners)) < 2:
        raise ValueError("At least two unseen writers are needed to make impostor trials")
    if method not in SCORING_METHODS:
        raise ValueError(f"Unknown scoring method: {method}")
    if len(set(enrollment_characters)) != len(enrollment_characters) or not enrollment_characters:
        raise ValueError("Enrollment character list must be nonempty and unique")
    enrollment_characters = set(enrollment_characters)
    query_characters = None if query_characters is None else set(query_characters)
    if query_characters is not None and query_characters & enrollment_characters:
        raise ValueError("Enrollment and query character sets overlap")
    if not set(row["subject"] for row in rows) <= set(owners):
        raise ValueError("Rows contain writers outside the requested cohort")
    trials = []
    for owner in owners:
        chosen = [i for i, row in enumerate(rows)
                  if row["subject"] == owner and row["character"] in enrollment_characters]
        if len(chosen) != len(enrollment_characters) or {
            rows[i]["character"] for i in chosen
        } != enrollment_characters:
            raise ValueError(f"Owner {owner} must have exactly one record per enrollment character")
        templates = embeddings[chosen]
        for i, row in enumerate(rows):
            if query_characters is None:
                if row["character"] in enrollment_characters:
                    continue
            elif row["character"] not in query_characters:
                continue
            is_genuine = row["subject"] == owner
            trials.append({
                "owner": owner,
                "query_subject": row["subject"],
                "character": row["character"],
                "path": row["path"],
                "genuine": int(is_genuine),
                "similarity": score_enrollment(embeddings[i], templates, method, top_k),
                "enrollment_count": len(chosen),
                "method": method,
                "top_k": top_k if method == "topk_mean" else "",
            })
    if not trials or len({t["genuine"] for t in trials}) != 2:
        raise ValueError("Need both genuine and impostor verification trials")
    return trials


def choose_far_threshold(trials, target_far):
    """Find lowest threshold giving empirical DEV FAR <= target_far.

    Strictly above the (k+1)-th largest impostor similarity leaves at most k
    falsely accepted impostor trials. Test data are never used here.
    """
    impostor = np.sort(np.array([r["similarity"] for r in trials if not r["genuine"]], dtype=float))[::-1]
    if len(impostor) == 0:
        raise ValueError("Development set has no impostor trials")
    allowed = int(np.floor(float(target_far) * len(impostor)))
    if allowed >= len(impostor):
        return -np.inf
    return float(np.nextafter(impostor[allowed], np.inf))


def score_trials(trials, threshold):
    if not trials:
        raise ValueError("Empty verification trials")
    scores = np.array([t["similarity"] for t in trials], dtype=float)
    y = np.array([t["genuine"] for t in trials], dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Need genuine and impostor trials for metrics")
    accepted = scores >= threshold
    n_gen = int(y.sum())
    n_imp = int(len(y) - n_gen)
    false_accept = int(np.sum(accepted & (y == 0)))
    false_reject = int(np.sum(~accepted & (y == 1)))
    fpr, tpr, _ = roc_curve(y, scores)
    index = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return {
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "far": false_accept / n_imp,
        "frr": false_reject / n_gen,
        "tar": 1 - false_reject / n_gen,
        "roc_auc": float(roc_auc_score(y, scores)),
        "eer_approx": float((fpr[index] + (1 - tpr[index])) / 2),
        "threshold": float(threshold),
    }
