import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_subject_variation import (
    EMF_NAMES,
    _focus_group_comparison,
    _pair_similarity,
    _prediction_analysis,
)


class SubjectAnalysisTest(unittest.TestCase):
    def test_emf_feature_name_count(self):
        self.assertEqual(len(EMF_NAMES), 58)

    def test_pair_similarity_uses_subject_character_units(self):
        rows = []
        for subject in ("A", "B", "C"):
            for character in ("一", "二", "三"):
                base = ord(subject) * 0.1
                char = ord(character) % 7
                rows.append({
                    "subject": subject,
                    "character": character,
                    "f1": base + char,
                    "f2": float(char),
                    "f3": float(ord(subject) % 5),
                })
        # A repeated trial should be averaged into the same subject-character unit.
        rows.append({
            "subject": "A", "character": "一",
            "f1": 99.0, "f2": 1.0, "f3": 1.0,
        })
        summary, per_subject = _pair_similarity(pd.DataFrame(rows), ["f1", "f2", "f3"])
        self.assertEqual(set(summary["pair_type"]), {
            "within_subject_different_character",
            "between_subject_same_character",
            "between_subject_different_character",
        })
        within = summary.set_index("pair_type").loc[
            "within_subject_different_character", "n_pairs"
        ]
        self.assertEqual(int(within), 9)
        self.assertEqual(len(per_subject), 3)

    def test_focus_group_comparison(self):
        df = pd.DataFrame([
            {"subject": "A", "split": "train", "acc_norm_rms": 1.0},
            {"subject": "B", "split": "train", "acc_norm_rms": 3.0},
            {"subject": "C", "split": "train", "acc_norm_rms": 5.0},
            {"subject": "A", "split": "test", "acc_norm_rms": 100.0},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            _focus_group_comparison(df, ["A", "B"], Path(tmp))
            out = pd.read_csv(Path(tmp) / "focus_group_vs_others_train.csv")
        row = out.loc[out["feature"] == "acc_norm_rms"].iloc[0]
        self.assertAlmostEqual(row["focus_mean_across_subjects"], 2.0)
        self.assertAlmostEqual(row["other_mean_across_subjects"], 5.0)

    def test_prediction_analysis_exports_errors_and_metrics(self):
        records = pd.DataFrame([
            {"path": "A/一.xlsx", "subject": "A", "character": "一",
             "split": "test", "duration_s": 1.0, "acc_norm_rms": 1.0},
            {"path": "B/一.xlsx", "subject": "B", "character": "一",
             "split": "test", "duration_s": 2.0, "acc_norm_rms": 2.0},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_path = root / "predictions.csv"
            pd.DataFrame([
                {"path": "A/一.xlsx", "true_subject": "A",
                 "predicted_subject": "A", "confidence": 0.9},
                {"path": "B/一.xlsx", "true_subject": "B",
                 "predicted_subject": "A", "confidence": 0.8},
            ]).to_csv(pred_path, index=False)
            merged, metrics, cm, labels = _prediction_analysis(
                pred_path, records, ["A", "B"], root
            )
            self.assertEqual(len(merged), 2)
            self.assertEqual(len(pd.read_csv(root / "error_samples.csv")), 1)
            self.assertEqual(cm.shape, (2, 2))
            self.assertEqual(labels, ["A", "B"])
            self.assertTrue((root / "per_subject_prediction_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
