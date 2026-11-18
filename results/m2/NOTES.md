# results/m2 — notes on how these results were produced

## v0 sweep (`*_v0.*` files)
- **Data:** `data/processed/{train,student_test}_candidates.parquet` built locally with M1's
  `scripts/build_dataset.py` (v1 feature set, 23 features from `artifacts/feature_list.json`).
  train: 2,354,439 rows / 80 trips / 208,417 GPS points (85.9 % valid, 2 trips flagged);
  student_test: 239,684 rows / 10 trips / 19,160 GPS points (99.3 % valid).
- **Subsampling:** `--obs-every 4` = every 4th GPS point per trip (all its candidates kept) for the training split.
  student_test is never subsampled. KNN is trained on ≤100k rows, RF / Extra Trees on ≤500k rows.
- **CV:** GroupKFold(5) by `trip_id` on `label_valid` rows of the 80 train trips. Seed 42.
- **Labels are provisional.** Ground-truth arcs are M1's fallback (`fallback_nearest_route_arc`: nearest arc among the
  reference-route arcs), because M3's order-preserving route alignment (`labels.py`) was not yet available.
  Every file records this in its `label_source` column/field.
  - Consequence: the true arc is always in the candidate set by construction, so **candidate miss rate is 0 by
    construction and is not reported** (blanked in the CSVs and model card).
  - When M3's alignment is pushed, the parquets are rebuilt and the sweep is re-run **before** final model selection
    and Optuna tuning. v0 numbers are for the first evaluation / early integration only.
- **Metrics:** from 12:30 on, pair metrics and Top-1 are computed with M3's shared `src/pathsnap/metrics.py`.
  Checked on the saved v0 student_test predictions: accuracy, precision, recall, F1, macro-F1, Top-1, confusion
  matrices and per-trip route continuity are identical to the earlier M2 implementation; only log loss differs for
  models that output exact 0/1 probabilities (GNB, KNN), because the team clips at 1e-9 (M2 previously 1e-7).
  v0 CSVs were produced with the old clipping.
- **Early export:** the best untuned v0 model is exported to `artifacts/` so M3 / M4 can integrate; it is replaced by the
  tuned model later (`artifacts/model_card.json` says which).

## v1 (`*_v1.*` files) — M3 labels
- **Labels:** M3's order-preserving route alignment (`src/pathsnap/labels.py`, `label_source = M3_alignment` for all 90
  trips). Parquets rebuilt with M1's `build_dataset.py` at 12:28 (same rows/candidates as v0; only labels changed).
  Ground-truth arc differs from the v0 fallback on 5.57 % of train points and 6.71 % of student_test points that are
  valid under both; valid share 85.8 % (train), 99.2 % (student_test). Candidate miss rate is meaningful again:
  0.0001 (train valid points), 0.0000 (student_test).
- **Optuna (`optuna_trials_v1.csv`, `optuna_best_params_v1.json`, `tuned_vs_untuned_v1.csv`):** LightGBM, TPE sampler
  (seed 42), median pruner, objective = mean 3-fold GroupKFold observation Top-1 on the train trips (78 with valid
  points — the 2 flagged offset trips have none; every 4th GPS
  point), 50 trials / 40 min cap. student_test is used only after the study, for the tuned-vs-untuned table.
- **Optuna result:** 50 trials (27 complete, 23 pruned), 1,354 s. Best = trial 41, CV Top-1 0.91860 (trial 5 had
  already reached 0.91858). Best params: num_leaves 185, learning_rate 0.0265, min_child_samples 19,
  colsample_bytree 0.839, subsample 0.745, reg_alpha 0.0019, reg_lambda 0.0033 (≤2000 trees, early stopping).
  Most important hyperparameter (fANOVA): colsample_bytree 0.47, then reg_alpha 0.18, min_child_samples 0.16.
- **Tuned vs untuned:** `tuned_vs_untuned_v1.csv` (same 3 folds and every-4th-point training rows for both) and
  `tuned_vs_untuned_fulldata_v1.csv` (both refit on all 2,064,331 valid train rows, scored on student_test).
  Tuning improves CV Top-1 by +0.30 pt and student_test log loss, macro-F1, undirected Top-1 and route continuity,
  but student_test Top-1 is flat (+0.03 pt subsampled, −0.11 pt full data) — within noise for 10 trips.
- **Exported:** `artifacts/model.joblib` = tuned LightGBM refit on all valid train rows (360 trees),
  student_test Top-1 0.9148, log loss 0.0364.
- **v1 sweep (`model_comparison_v1.csv`, `metrics_summary_v1.csv`, `cv_folds_v1.csv`):** 9 models, 5-fold GroupKFold,
  every 4th GPS point, M3 labels, M3 metrics. Top four are tied on CV Top-1 (hgb 0.9138, lgbm 0.9137, rf 0.9136,
  xgb 0.9130; per-fold head-to-head lgbm vs hgb 2/5, vs rf 2/5, vs xgb 3/5). Selection rule tie-break → LightGBM has
  the lowest CV log loss (0.0407 vs 0.0410 / 0.0419 / 0.0412), the smallest train–CV gap among the four
  (0.029 vs 0.045 hgb, 0.057 rf) and is faster than hgb/rf. **LightGBM confirmed** → tuned LightGBM stays in `artifacts/`.
  student_test is not used for selection (hgb has the highest ST Top-1, 0.9185 vs 0.9162).
