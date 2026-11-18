# 2. Dataset and Exploratory Data Analysis

> Drafting notes (delete before submitting). Numbers marked **[FILL]** come from CSVs that only exist on your
> machine (`candidate_stats.csv`, `cands_per_obs_hist.csv`, `offset_trips_by_rule.csv`). Everything else was
> computed directly from the raw `.track/.nodes/.arcs/.route` files of the organiser package. Those raw-file numbers
> use a simple local metric approximation, so if `results/m1/*.csv` (UTM-based) differs in the last digit, trust the CSV.
> Rewrite sentences in your own voice and make sure you can explain each one in the viva.

## 2.1 Data inventory and partitions

We used only the organiser package. It has two partitions:

| Partition | Trips | GPS points | Points per trip (min / median / max) | Nodes per trip | Arcs per trip |
|---|---|---|---|---|---|
| `train` | 80 | 208,417 | 361 / 1,529 / 19,313 | 299 – 155,003 | 649 – 307,392 |
| `student_test` | 10 | 19,160 | 262 / 1,420 / 3,958 | 1,770 – 26,635 | 3,577 – 56,027 |

(source: `results/m1/inventory_summary.csv`, `trip_level_stats.csv`; figure `../../results/m1/network_size.png`)

Each trip has four whitespace-separated files without headers and with Windows (CRLF) line endings:
`.track` (lon, lat, t in seconds, **longitude first**), `.nodes` (lon, lat; row number = node id),
`.arcs` (from_node, to_node; row number = arc id; arcs are **directed**) and `.route` (the ordered arcs the vehicle
traversed). The route has one entry per arc travelled, not one per GPS point, so it cannot be used directly as a
per-point label. The road graph is different for every trip and is very large for some trips, which is why we
needed a spatial index (an STRtree) instead of comparing every point with every arc.

**Partition used:** the inventory, quality, network and route analyses below were run on **train (80 trips)** and
repeated on **student_test (10 trips)** for comparison. The candidate-radius study (2.7) used **train only**.
The evaluator trips are not in our package and were never touched.

## 2.2 Preprocessing

1. **Parsing.** `pd.read_csv(sep=r"\s+", header=None)` handles the CRLF endings and the multiple spaces.
   `load_trip(..., with_route=False)` is the default so prediction code can never read the ground truth.
2. **Projection.** Each trip is projected to the UTM zone of its own centroid (`geo.py`, EPSG 326xx/327xx), so every distance
   is in metres. Raw degrees would be wrong because one degree of longitude shrinks with cos(latitude) and our trips
   span latitudes from −34.6° to 61.2° (see 2.4).
3. **Time handling.** Timestamps are checked, gaps (dt > 10 s) are marked, and heading/speed/window features never use
   points across a gap.
4. **Candidate generation.** For each GPS point we keep every arc within R metres plus always the 8 nearest arcs,
   capped at 30 (2.7).
5. **Labels.** `.route` is aligned to the track to get one arc per point. Until the alignment code (`labels.py`) was
   merged, the build used a fallback: the nearest arc among the route arcs. Both trip summaries I checked
   report `label_source = fallback_nearest_route_arc`. **[FILL: update this sentence if you reran after `labels.py` landed.]**
6. **Validity.** An observation is a valid training label only if its label distance is within the threshold and its
   trip is not flagged as an offset trip (2.5).

## 2.3 Sampling and data quality

| Check | train (80 trips) | student_test (10 trips) |
|---|---|---|
| NaN values in `.track` | 0 | 0 |
| Duplicate or non-increasing timestamps | 0 | 0 |
| Average share of exactly 1 s steps per trip | 93.8 % | 98.2 % |
| Trips with a gap > 10 s | 40 | 2 |
| Trips with a gap > 60 s | 22 | 1 |
| Longest gap | 153,064 s (about 1.8 days) | 210 s |
| Steps with implied speed > 60 m/s | 536 | 2 |

(source: `results/m1/quality_summary.csv`; figure `../../results/m1/dt_hist.png`, log y-axis)

Timestamps are clean (no NaNs, strictly increasing), so no rows had to be dropped. The sampling is not uniform
though: the data is mostly 1 Hz, but half of the train trips contain at least one gap longer than 10 s and the
largest gap is days long. A speed or heading computed across such a gap is meaningless, so I mark the point with
`gap_flag` and keep windows from crossing it. The implied-speed outliers are rare (536 steps out of 208,417, about 0.26 %), so I flagged
them instead of deleting them.

## 2.4 Geography and projection

Trips come from all over the world: centroid latitude −34.6° to 61.2° and longitude −100.3° to 137.2° in train
(`../../results/m1/world_centroids.png`). So absolute coordinates are useless as model features (they would not
generalise to unseen places), and distances must be computed in a metric projection per trip, not in degrees.

## 2.5 Road network and reference routes

**Road network (train).**
- About **92 %** of arcs have a reverse twin with the same geometry (mean over trips 0.922, lowest trip 0.351; student_test 0.927).
  A two-way road therefore appears as two arcs at the same place, and distance alone cannot tell them apart. That is
  why heading is the most important idea in the features.
- No isolated nodes. 79 of 80 train graphs have more than one weakly connected component and all 80 have more than one
  strongly connected component (one-way streets, dead ends, clipped areas), so routes must not be assumed to be reachable from everywhere.
- 100 zero-length arcs in train (6 in student_test): two nodes with identical coordinates.
- Typical arc length: the median over trips of the per-trip median is about 24.5 m and of the per-trip 95th percentile about 121 m (student_test 31.8 m and 146 m).

**Reference routes.**
- 10 of 80 train trips have a break in the route (13 breaks in total, i.e. consecutive arcs that do not connect);
  9 trips revisit at least one arc (loops). In student_test, 1 trip has a break and 1 has a repeated arc.
- Total route length is about 2,222 km in train and 247 km in student_test.
- The distance from GPS points to the nearest route arc is small for most trips: the median over trips of the
  per-trip median is about 4.6 m (student_test 4.7 m). This is the label-independent quantity used to detect offset trips.
  (`../../results/m1/gps_to_route_boxplot_train.png`)

**Offset trips.** The rule is: a trip is flagged if its *median* GPS-to-route distance is above 50 m. This rule needs no
labels and no hand-picking. In train it flags exactly two trips, **00000033** and **00000054**:

| Trip | Points | Median GPS→route distance | 95th percentile |
|---|---|---|---|
| 00000033 | 5,844 | about 122 m | about 141 m |
| 00000054 | 19,313 | about 137 m | about 209 m |

**[FILL: replace the four distance values with the numbers in `results/m1/offset_trips_by_rule.csv`.]**
No student_test trip is flagged (largest median 6.5 m). The next-worst train trip, 00000043, has a median of only about 11 m
(p95 about 72 m), so the 50 m threshold separates the two offset trips from the rest with a very large margin.

Why I think this is a data issue and not a model issue: the whole trip is shifted, not a few points. The median itself is above 100 m,
whereas every other trip is mostly below 10 m. A trip that is constantly 120–140 m from its own reference route cannot
be a matching mistake. The decision: label alignment for these trips is not reliable, so they are marked
`label_valid = False` (excluded from training and from the metrics) but the model still predicts them.

## 2.6 Labels: why the nearest arc is not enough

With the fallback labels, the plain nearest arc disagrees with the label in **14–58 %** of observations per trip (directed comparison) across the 30 trip summaries
I checked (20 train, 10 student_test). If the reverse twin is accepted as correct (undirected), this drops to **2–32 %**.
The gap between the two numbers is the twin-arc problem from 2.5. The remaining disagreement is at junctions and parallel roads.
So the nearest arc is a poor classifier and a poor label, which is why we use a learned classifier on top of candidates and an HMM
(order-aware decoding) afterwards. (`results/m1/trip_summary_<split>.csv`, columns `nearest_arc_disagree` and `nearest_arc_disagree_undirected`.)
Note that these disagreement numbers were measured against the fallback labels.

## 2.7 Candidate generation

Rule: all arcs within R = 50 m of the point, plus the 8 nearest arcs even if they are farther away, at most 30 per point.
The k-nearest part guarantees that a point far from the road still receives candidates; across the 30 trip summaries every
trip has `cand_miss_rate = 0.0` and the average number of candidates per observation is between 8.3 and 21.1.

**[FILL from `candidate_stats.csv` (train only): a small table of miss rate and mean candidates per observation for R = 20, 30, 50, 75, 100,
and one sentence on why 50 m is the compromise: larger R lowers the miss rate but multiplies the rows the classifier has to score.]**
Figure: `../../results/m1/miss_rate_vs_radius.png`.

## 2.8 Features and why

23 features per (observation, candidate) pair, listed in `artifacts/feature_list.json`. No absolute coordinates are used.
Any learned preprocessing is fitted on train only, and cross-validation is grouped by trip (GroupKFold) because points of one trip are strongly correlated.

| Group | Features | Why |
|---|---|---|
| Distance | `dist_m`, `dist_rank`, `dist_minus_min`, `dist_ratio`, `dist_z`, `n_cands` | The true arc is usually the closest one. Relative versions compare candidates within the same observation, and `dist_z` divides by a per-trip noise estimate (label-free) so noisy and clean trips are comparable. |
| Direction | `heading_cos`, `heading_sin_abs`, `speed_mps`, `has_twin` | `heading_cos` is +1 for the same direction and −1 for the reverse twin, which separates the two arcs of a two-way road. Heading is unreliable when slow, so speed lets the model learn that. |
| Arc geometry | `arc_len_m`, `proj_frac`, `dist_to_end_m`, `from_deg`, `to_deg` | Short arcs and complex intersections behave differently, and the position along the arc tells whether the vehicle is about to turn. |
| Temporal consistency | `win2_mean_dist`, `win5_mean_dist`, `win5_rel`, `win5_heading_cos` | The true arc stays close for the whole window of neighbouring points, while wrong arcs drift away. This damps single-point noise. |
| Sampling | `dt_prev`, `dt_next`, `gap_flag`, `accel` | Gaps weaken window and heading features. Acceleration changes tend to occur at stops and junctions. |

Heading is set to 0 (unknown) when the ±2-point displacement is below 2 m.

## 2.9 Data-quality issues, assumptions and decisions

| Issue | Evidence | Decision |
|---|---|---|
| CRLF line endings, whitespace separators | all four file types | Parse with `sep=r"\s+"`. |
| Gaps in time | 40 / 80 train trips have a gap > 10 s, 22 have > 60 s, longest 153,064 s | `gap_flag`; windows and heading do not cross a gap. |
| Implied-speed outliers | 536 steps > 60 m/s in train | Flagged, not deleted. |
| Offset trips | 00000033 and 00000054, median GPS→route distance far above the 50 m rule | `label_valid = False`, excluded from training and metrics, still predicted. |
| Route breaks and loops | 10 train trips with breaks (13 in total), 9 with repeated arcs | Kept; the HMM and the metrics must tolerate non-simple routes. |
| Twin arcs | about 92 % of arcs | Heading features; report directed and undirected accuracy. |
| Worldwide extent | latitude −34.6° to 61.2° | Per-trip UTM projection; no absolute coordinates as features. |
| Fallback labels | `label_source = fallback_nearest_route_arc` | Replace by order-preserving alignment (M3); report which was used. |

## 2.10 Map

The interactive map of trip 00000046 is `../../results/maps/eda_00000046.html` (use the PNG `../../results/maps/eda_00000046.png`
in the PDF; it is produced when `make_maps.py` is run with `--png`). It shows the road network in grey, raw GPS points in blue and the
reference route in green. **[FILL: 2 sentences about what you see in the map, for example how closely the GPS follows the green route and where it deviates.]**
Trip 00000046 has a small network (649 arcs) and a 95th percentile GPS→route distance of about 21 m, which is on the higher side for the trips in our summaries.
