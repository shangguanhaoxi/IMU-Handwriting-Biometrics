"""Synthetic regressions for writer-disjoint enrollment and thresholding."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from verification.protocol import (
    choose_far_threshold, create_trials, load_protocol, normalize_rows,
    score_trials, score_enrollment,
)


class VerificationProtocolTests(unittest.TestCase):
    def test_writer_disjoint_protocol_rejects_overlap(self):
        p = {
            "train_subjects": ["A", "F"],
            "dev_subjects": ["lrx", "sg"],
            "test_subjects": ["yl", "ypb"],
            "enrollment_characters": ["一"],
            "query_policy": "all_characters_not_enrolled",
            "unseen_test_characters": ["十"],
            "target_dev_far": .05,
        }
        cfg = {"train_characters": ["一", "人"], "test_characters": ["十"]}
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "protocol.json"
            file.write_text(json.dumps(p), encoding="utf-8")
            self.assertEqual(
                load_protocol(file, ["A", "F", "lrx", "sg", "yl", "ypb"], cfg),
                p,
            )
            p["dev_subjects"] = ["A", "sg"]
            file.write_text(json.dumps(p), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_protocol(file, ["A", "F", "lrx", "sg", "yl", "ypb"], cfg)

    def test_enrollment_queries_are_disjoint(self):
        rows = [
            {"subject": subject, "character": char, "path": f"{subject}/{char}.xlsx"}
            for subject in ("B", "C")
            for char in ("一", "人", "十")
        ]
        emb = np.array([
            [1., 0.], [1., 0.], [1., .2],
            [0., 1.], [0., 1.], [.1, 1.],
        ])
        trials = create_trials(rows, emb, ["B", "C"], ["一", "人"])
        self.assertEqual(len(trials), 4)
        self.assertEqual({t["character"] for t in trials}, {"十"})
        self.assertEqual(sum(t["genuine"] for t in trials), 2)
        self.assertGreater(
            next(t["similarity"] for t in trials if t["owner"] == "B" and t["genuine"]),
            next(t["similarity"] for t in trials if t["owner"] == "B" and not t["genuine"]),
        )
        with self.assertRaises(ValueError):
            create_trials(rows, emb, ["B", "C"], ["一"], ["一", "十"])

    def test_three_scoring_rules_with_ten_templates(self):
        # 3 close templates and 7 orthogonal templates give different scores.
        query = np.array([1., 0.])
        templates = np.vstack([
            np.tile(np.array([[1., 0.]]), (3, 1)),
            np.tile(np.array([[0., 1.]]), (7, 1)),
        ])
        s_mean = score_enrollment(query, templates, "mean_prototype")
        s_max = score_enrollment(query, templates, "max_similarity")
        s_top3 = score_enrollment(query, templates, "topk_mean", 3)
        self.assertAlmostEqual(s_max, 1.)
        self.assertAlmostEqual(s_top3, 1.)
        self.assertAlmostEqual(s_mean, 3. / np.sqrt(58.))
        with self.assertRaises(ValueError):
            score_enrollment(query, templates, "topk_mean", 11)

    def test_k10_trial_alignment_across_methods(self):
        chars = [f"char{i}" for i in range(10)]
        rows = [
            {"subject": owner, "character": char, "path": owner + "/" + char}
            for owner in ("B", "C")
            for char in chars + ["queryA", "queryB"]
        ]
        emb = np.array([
            [1., 0.] if r["subject"] == "B" else [0., 1.]
            for r in rows
        ])
        trials_by_method = {
            method: create_trials(
                rows, emb, ["B", "C"], chars, method=method, top_k=3
            )
            for method in ("mean_prototype", "max_similarity", "topk_mean")
        }
        for trials in trials_by_method.values():
            self.assertEqual(len(trials), 8)
            self.assertEqual({r["character"] for r in trials}, {"queryA", "queryB"})
            self.assertEqual({r["enrollment_count"] for r in trials}, {10})
        paths = [
            {(r["owner"], r["query_subject"], r["path"]) for r in trials}
            for trials in trials_by_method.values()
        ]
        self.assertEqual(paths[0], paths[1])
        self.assertEqual(paths[1], paths[2])

    def test_reject_duplicate_enrollment_trial(self):
        rows = [
            {"subject": subject, "character": char, "path": subject + "/" + char}
            for subject in ("B", "C") for char in ("一", "十")
        ]
        rows.append({"subject": "B", "character": "一", "path": "B/一_trial2"})
        emb = np.ones((len(rows), 2))
        with self.assertRaisesRegex(ValueError, "exactly one record"):
            create_trials(rows, emb, ["B", "C"], ["一"])

    def test_far_calibration_does_not_use_genuine_scores(self):
        dev = (
            [{"similarity": s, "genuine": 0} for s in [.1, .2, .3, .4, .5]]
            + [{"similarity": s, "genuine": 1} for s in [.6, .7]]
        )
        threshold = choose_far_threshold(dev, 0.0)
        self.assertGreater(threshold, 0.5)
        self.assertEqual(score_trials(dev, threshold)["false_accept"], 0)
        new_genuine = dev[:5] + [{"similarity": .01, "genuine": 1}]
        self.assertEqual(threshold, choose_far_threshold(new_genuine, 0.0))

    def test_score_metrics(self):
        trials = [
            {"similarity": .9, "genuine": 1},
            {"similarity": .8, "genuine": 1},
            {"similarity": .2, "genuine": 0},
            {"similarity": .1, "genuine": 0},
        ]
        m = score_trials(trials, 0.5)
        self.assertEqual(m["far"], 0.)
        self.assertEqual(m["frr"], 0.)
        self.assertEqual(m["roc_auc"], 1.)

    def test_normalize_rows(self):
        v = normalize_rows(np.array([3., 4.]))
        self.assertEqual(v.shape, (2,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.)


if __name__ == "__main__":
    unittest.main()
