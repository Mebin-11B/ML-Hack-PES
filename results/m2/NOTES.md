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
- **Early export:** the best untuned v0 model is exported to `artifacts/` so M3 / M4 can integrate; it is replaced by the
  tuned model later (`artifacts/model_card.json` says which).
