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
* run grid G ∈ {12,24,48} × ε × α on CLINC150 and BFCL — done (make calibrate, 362 files, no errors)
* verify empirical survival ≥ 1 − α on test split — done: 45 certified picks, 1 miss (kev-4b bfcl G=48 α=0.05: 0.945)
* test transfer: G, R chosen on one dataset, evaluated on others — done from tournament grid: G=24 best for both datasets, permute best for Kev on both
* test transfer: ε calibrated on one dataset, evaluated on another — done (cmd/eval-transfer): 23/46 certified ε hold on the other dataset; towards easier task yes, towards CLINC150 no
* run LLM baseline (qwen-7b) — done: CLINC150 k=150 0.792 / 939 ms, BFCL k=255 0.733 / 7.6 s
* generate plots — done (cmd/plots, 40 pgfplots figures)
* generate result tables — done (paper/*.gen.tex)
* write article — full draft in paper/main.tex (29 pages, 83 cited references, 50 figures and tables); to do: proofread, abstract polish, venue formatting
* clean latency for CLINC150 rows (one model, one worker) — optional
* copy pilot scripts into pilot/ — pending
