## Milestones

* create Dockerfile (llama.cpp CUDA arm64)
* build image on GX10
* download Laya BF16 GGUF
* download Kev-4B BF16 GGUF
* make Laya quantizations (Q8_0, Q6_K, Q4_K_M, Q4_0, Q2_K) with llama-quantize
* create docker-compose.yaml (one llama-server per model/quantization)
* run docker-compose on GX10 with GPU
* test if models are working using tests.http file
* verify Laya llama.cpp BF16 vs original PyTorch (Python script, 200–500 examples)
* verify Kev-4B llama.cpp BF16 vs original PEFT (Python script, 200–500 examples)
* download datasets (MASSIVE, XNLI, DAIR Emotion, Banking77, AG News, SST-5)
* split datasets into calibration / test
* init Go module
* implement loading dataset in Go
* implement dataset → /v1/systemone question mapping in Go
* implement Laya/Kev-4B client in Go
* implement saving full probabilities per example/model/quantization
* run hypothesis test: Laya BF16 vs Q4_K_M on one dataset, flips vs margin
* go / no-go decision
* literature review (cascades with conformal guarantees, speculative decoding)
* implement quality metrics in Go (accuracy, macro-F1, ECE, Brier, NLL)
* implement agreement metrics in Go (flip rate, flip rate vs margin)
* implement latency measurement in Go (p50/p95/p99, throughput)
* implement energy measurement (GPU power → J/decision)
* run Laya vs Kev-4B comparison (all datasets)
* run Laya quantization benchmark (all quantizations, all datasets)
* choose Laya quantization sweet spot
* implement cascade in Go (margin, τ, escalation to Kev-4B)
* implement conformal τ selection on calibration split
* run cascade evaluation on test split
* verify empirical agreement ≥ 1 − α
* generate plots (Pareto front, flip vs margin, reliability diagrams)
* generate result tables
* writing article
