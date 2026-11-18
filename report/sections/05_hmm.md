# 5. HMM map matching

## Decoder design

1. **Hidden states:** each observation's candidate directed road arcs. State indices refer to candidate rows; decoded indices map back to the corresponding `arc_id`.
2. **Emissions:** classifier probabilities are normalized within each observation and converted to log space as `log(p + 1e-9) / T`. When no model artifact is available, the predictor uses the documented distance-only Gaussian emission with σ = 10 m.
3. **Transitions:** each previous/current candidate pair gets a network distance. On one arc, that is the forward along-arc displacement, allowing up to 15 m backwards for GPS noise. Across arcs it is the remaining distance on the old arc, a directed shortest path between nodes, then the offset along the new arc. We score `-|d_net - d_gc| / β`; transitions beyond `d_gc + max(D0, v_max·dt)` or with unreachable graph nodes are impossible (`-∞`).
4. **Missing candidates and gaps:** the candidate generator guarantees at least one candidate per GPS fix. For any omitted fix, decoding leaves `-1` and bridges elapsed time from the previous candidate observation. A time gap above `gap_s` or a step with no feasible transition starts a fresh Viterbi segment from emissions. Route output collapses repeated arcs and inserts directed shortest-path arcs where adjacent decoded arcs do not connect.
5. **Disconnected segments:** if every transition is impossible, the decoder restarts at the current observation using its emissions, preserving an output instead of inventing an unreachable path. The output sequence can still contain breaks where the graph has no connecting path.
6. **Did the HMM help?** On the 10 `student_test` trips, mean directed top-1 rose from 90.2% (classifier only) to 92.7%; mean route F1 rose from 70.0% to 97.4%; mean continuity rose from 73.8% to 91.2%; and mean breaks fell from 122.3 to 32.2. Nearest-arc top-1 was 56.5%. Decoder parameters were selected on three train trips (T=2, β=15, gap=30 s); the same setting was confirmed on all 10 held-out student test trips. These measured values are in `results/m3/clf_vs_hmm.csv` and `results/m3/decoder_tuning.csv`.

The labeler aligns route **positions**, rather than unique arc IDs, to preserve repeated arcs and loops. Its transition is monotone and penalizes skipped route positions; point-to-segment distance supplies the observation cost. Sanity distances across all 80 train trips are recorded in `results/m3/label_sanity.csv`.

Map examples selected by measured change in top-1 are in `results/maps/`: `hmm_helps_*.html` and `hmm_fails_*.html`. The latter illustrates that continuity constraints can still choose a plausible connected road when the GPS/candidate evidence is ambiguous.
