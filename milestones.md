## Milestones

* copy pilot scripts and outputs from scratchpad into repo (`pilot/`)
* download Kev-0.8B and Kev-9B BF16 GGUF (see README)
* init Go module
* implement dataset loading in Go
* implement /v1/systemone client in Go (save full probabilities)
* implement baseline: single call with all options
* run diagnosis: accuracy / ECE / flip rate vs number of options (k = 3, 6, 12, 24, 59, 77, 150)
* run diagnosis: accuracy / flip rate vs option length (tokens) and with/without descriptions
* run diagnosis: flip rate on reversed and shuffled order
* implement permutation averaging (R shuffles, mean probabilities)
* implement fixed tournament (groups of G, winner per group, final)
* implement adaptive tournament (threshold ε, all options with p ≥ ε pass, repeat until ≤ G, final)
* implement conformal selection of ε on calibration split (gold survives all rounds ≥ 1 − α)
* implement label-free ε variant (agreement with expensive tournament instead of gold)
* implement metrics in Go (accuracy, macro-F1, ECE, Brier, flip rate, survival rate)
* implement cost measurement in Go (calls per decision, latency p50/p95, J/decision)
* run grid: G × R × α for Laya and Kev on all datasets
* verify empirical survival ≥ 1 − α on test split
* test transfer: G, R chosen on one dataset, evaluated on others
* test transfer: ε calibrated on one dataset, evaluated on another (expect failure, measure size)
* run LLM baseline (qwen-7b service in compose) for accuracy and latency reference
* generate plots (accuracy/ECE vs k, Pareto cost vs accuracy, survival vs α, flip vs R)
* generate result tables
* write article
