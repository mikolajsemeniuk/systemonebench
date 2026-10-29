## Milestones

* copy pilot scripts and outputs from scratchpad into repo (`pilot/`)
* run diagnose-k on all models (make diagnose-k)
* run tournament grid on all models (make tournament) — done; rerun 8 Kev BFCL files with -workers 1 (timeouts)
* download Kev-0.8B and Kev-9B BF16 GGUF (see README)
* diagnosis with/without descriptions — done (-nodesc): Laya BFCL single identical (budget truncates names, not only descriptions); tournament g8 0.587 -> 0.632; still far below Kev because per-round errors compound (k=12 acc 0.87-0.91, 3 rounds)
* conformal selection of ε — done: pkg/conformal (pick on half A, certify on half B; survival is not monotone in ε), cmd/calibrate, cmd/eval-calibrate; make calibrate running (CLINC150 done, BFCL pending)
* chain variant (final p × product of round p) — done; on CLINC150 Kev still below single (kev-9b 0.762 g24 vs 0.792), Laya unchanged; Kev needs permute, not tournament
* implement label-free ε variant (agreement with expensive tournament instead of gold)
* implement survival rate metric (gold survives tournament) — done: pkg/metrics/survival.go
* implement cost measurement in Go (calls per decision, latency p50/p95, J/decision)
* run grid: G × R × α for Laya and Kev on all datasets
* verify empirical survival ≥ 1 − α on test split
* test transfer: G, R chosen on one dataset, evaluated on others
* test transfer: ε calibrated on one dataset, evaluated on another (expect failure, measure size)
* run LLM baseline (qwen-7b service in compose) for accuracy and latency reference
* generate plots (accuracy/ECE vs k, Pareto cost vs accuracy, survival vs α, flip vs R)
* generate result tables
* write article
