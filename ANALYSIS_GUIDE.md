# Subject-specific IMU motion diagnostics (9 unique writers)

**Identity correction (2026-11-22):** Historical folders B/C/D/E were aliases of the same physical writer as A. Only A is retained; current active writers are A, F, lrx, sg, yl, ypb, zjl, zjq, zpc. Do not draw distinct-identity conclusions from older 13-class diagnostic outputs.

This analysis script is for the current six-axis OMC experiment. It reads the
already-extracted data directory and writes descriptive reports; it does not
train a model, alter checkpoints, or change the feature cache.

## Run on AutoDL

After syncing main:

~~~bash
cd /root/autodl-tmp/OMC_IMU_Research

python analyze_subject_variation.py \
  --data_path ../excel \
  --split_config imu_split_config.json \
  --output_dir analysis/subject_diagnostics \
  --predictions runs/omc_9subject_vat/evaluation_test/predictions.csv
~~~

If the test predictions are stored elsewhere, pass their actual path to
--predictions. The predictions argument is optional; without it, the script
still analyzes raw motion statistics, EMF features, and cross-character
similarity.

The defaults assume the current experiment settings: six-axis input, 100 Hz
fallback sampling rate for XLSX files, sequence length 256, and minimum length
16. If the checkpoint was trained with different settings, pass the matching
--sample_rate, --seq_len, and --min_length values.

## What is analyzed

1. **Per-record raw motion statistics**: axis-wise mean, standard deviation,
   RMS and percentiles for gravity-removed linear acceleration and gyroscope;
   vector-magnitude statistics; sample count; duration; effective sampling
   rate; timestamp interval variability; and the 3-by-3 acceleration/gyro
   Pearson correlation matrix.
2. **Velocity proxy**: acceleration is integrated using the same zero-initial
   velocity and no-detrending convention used by the current EMF pipeline.
   This is only an integrated-acceleration proxy. It is not ground-truth pen
   velocity and is sensitive to bias and drift.
3. **EMF summaries**: the 58 global EMF features and mean, standard deviation,
   and RMS of each of the 20 EMF sequence channels.
4. **Subject and split summaries**: per-subject statistics separately for
   train, validation, and test.
5. **Focus writer versus other subjects**: a descriptive comparison using equal-weight
   subject-level means from the training split. Change the group with
   --focus_subjects A. This is not a significance test.
6. **Cross-character feature similarity**: by default, use training records
   only. Repeated trials are first averaged within each subject-character pair.
   It compares (a) the same person's different characters, (b) different
   people writing the same character, and (c) different people writing
   different characters. Similarities use standardized handcrafted EMF
   summaries; they are not OMC learned embeddings.
7. **Test error backtracking**: when --predictions is provided, the script
   exports per-subject precision/recall/F1, a labeled confusion matrix,
   true-to-predicted confusion pairs, individual error records with motion
   features, and feature summaries for correct versus incorrect test records.
   If VAT columns are present in predictions.csv, it also exports overall,
   per-class, and per-subject VAT speed-class metrics and a VAT confusion matrix.

## Main outputs

- record_level_features.csv: one row per successfully parsed recording.
- subject_split_summary.csv: per-subject, per-split means and standard
  deviations.
- focus_group_vs_others_train.csv: selected writer(s) versus the remaining subjects,
  based on equal-weight subject averages from training data.
- feature_similarity_pair_summary.csv: group-level pairwise similarity.
- per_subject_cross_character_similarity.csv: per-subject similarity.
- per_subject_prediction_metrics.csv: test metrics per identity, if
  predictions were supplied.
- error_samples.csv and confusion_pairs.csv: test error trace, if
  predictions were supplied.
- vat_metrics.json, vat_per_class_metrics.csv, vat_per_subject_metrics.csv,
  and vat_confusion_matrix.csv: VAT diagnostics when velocity predictions exist.
- PNG plots for motion magnitude, speed proxy versus duration, similarity
  groups, test accuracy, and test confusion matrix when matplotlib is installed.
- analysis_summary.json and analysis_skipped_records.csv: run metadata
  and records that could not be parsed.

## Interpretation cautions

- Do not interpret the integrated-acceleration speed proxy as true writing
  speed. Bias, initial velocity, timestamp quality, and integration drift can
  affect it.
- Do not infer a causal explanation from the focus-writer group comparison. It is
  descriptive and should be followed by per-record inspection and repeated
  experiments.
- Cross-character similarity is computed from handcrafted EMF summaries, not
  learned OMC embeddings. It is a diagnostic of input-feature stability, not
  a direct measurement of the classifier's embedding geometry.
- The primary similarity analysis uses the training split to avoid using test
  labels to guide model development. The optional prediction error analysis
  reads test predictions only after the checkpoint has been selected.
- Raw acceleration and gyroscope units are not converted. Confirm the units
  and sensor-axis conventions from the acquisition protocol before comparing
  absolute magnitudes across subjects or collection sessions.
- The script parses the spreadsheets again and may take a few minutes. It does
  not overwrite the model outputs or training cache.
