# PathSnap: Distributed Map-Matching & Route Reconstruction

**UE24CS352A Machine Learning: Hackathon Technical Report**

**Repository:** https://github.com/Mebin-11B/ML-Hack-PES

| Member | Role |
|---|---|
| Team Member 1 | Data processing, EDA, candidate generation, feature engineering, Folium EDA maps |
| Team Member 2 | Model experimentation (9 models), model selection, Bayesian hyperparameter optimisation |
| Team Member 3 | Ground-truth route alignment, metrics, HMM/Viterbi decoding, prediction pipeline, scoring |
| Team Member 4 (lead) | Integration, FastAPI backend, evaluator UI, ZeroTier deployment, report assembly |

## 1. Overview

Each GPS observation is matched to a directed road segment (arc) of the trip's road network in four steps:

1. **Candidates:** arcs within 50 m, plus the 8 nearest arcs, up to 30 per observation.
2. **Classifier:** a tuned LightGBM classifier scores each (observation, candidate) pair.
3. **HMM:** a hidden Markov model with Viterbi decoding chooses the most likely connected sequence of arcs, using road-network distances.
4. **Serving:** a FastAPI server on Laptop 1 runs this pipeline for any evaluation trip and scores it server-side; a UI on Laptop 2 displays the result. The two laptops are connected over ZeroTier.
