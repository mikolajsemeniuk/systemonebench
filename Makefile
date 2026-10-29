# Reproduce the experiment step by step: make models up data baseline eval
# Each step writes files; cmd/* -> output/ -> cmd/eval-* -> paper/*.gen.tex

.PHONY: all models up down data baseline diagnose-k tournament calibrate eval test clean

all: data baseline diagnose-k tournament calibrate eval

# 1. download GGUF models into models/
models:
	mkdir -p models
	curl -L -o models/Laya-BF16.gguf https://huggingface.co/ggml-org/Laya-GGUF/resolve/main/Laya-BF16.gguf
	curl -L -o models/Kev-0.8B-BF16.gguf https://huggingface.co/ggml-org/Kev-0.8B-GGUF/resolve/main/Kev-0.8B-BF16.gguf
	curl -L -o models/Kev-4B-BF16.gguf https://huggingface.co/ggml-org/Kev-4B-GGUF/resolve/main/Kev-4B-BF16.gguf
	curl -L -o models/Kev-9B-BF16.gguf https://huggingface.co/ggml-org/Kev-9B-GGUF/resolve/main/Kev-9B-BF16.gguf
	curl -L -o models/Qwen2.5-7B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# 2. start one llama-server per model (GPU)
up:
	docker compose up -d --build
	for p in 8081 8082 8083 8084 8090; do until curl -sf localhost:$$p/health >/dev/null; do sleep 2; done; echo "port $$p ok"; done

down:
	docker compose down

# 3. download datasets into pkg/datasets (embedded into the Go binaries; re-running changes the random split)
data:
	go run ./cmd/datasets

# 4. baseline: one call per example with all options (test split)
baseline:
	go run ./cmd/baseline -model laya -url http://localhost:8081 -dataset emotion
	go run ./cmd/baseline -model laya -url http://localhost:8081 -dataset xnli
	go run ./cmd/baseline -model laya -url http://localhost:8081 -dataset massive
	go run ./cmd/baseline -model laya -url http://localhost:8081 -dataset banking77
	go run ./cmd/baseline -model laya -url http://localhost:8081 -dataset clinc150
	go run ./cmd/baseline -model kev-0.8b -url http://localhost:8082 -dataset emotion
	go run ./cmd/baseline -model kev-0.8b -url http://localhost:8082 -dataset xnli
	go run ./cmd/baseline -model kev-0.8b -url http://localhost:8082 -dataset massive
	go run ./cmd/baseline -model kev-0.8b -url http://localhost:8082 -dataset banking77
	go run ./cmd/baseline -model kev-0.8b -url http://localhost:8082 -dataset clinc150
	go run ./cmd/baseline -model kev-4b -url http://localhost:8083 -dataset emotion
	go run ./cmd/baseline -model kev-4b -url http://localhost:8083 -dataset xnli
	go run ./cmd/baseline -model kev-4b -url http://localhost:8083 -dataset massive
	go run ./cmd/baseline -model kev-4b -url http://localhost:8083 -dataset banking77
	go run ./cmd/baseline -model kev-4b -url http://localhost:8083 -dataset clinc150
	go run ./cmd/baseline -model kev-9b -url http://localhost:8084 -dataset emotion
	go run ./cmd/baseline -model kev-9b -url http://localhost:8084 -dataset xnli
	go run ./cmd/baseline -model kev-9b -url http://localhost:8084 -dataset massive
	go run ./cmd/baseline -model kev-9b -url http://localhost:8084 -dataset banking77
	go run ./cmd/baseline -model kev-9b -url http://localhost:8084 -dataset clinc150

# 5. diagnosis: k options (gold + random distractors), sorted and reversed order
diagnose-k:
	go run ./cmd/diagnose-k -model laya -url http://localhost:8081 -dataset clinc150
	go run ./cmd/diagnose-k -model laya -url http://localhost:8081 -dataset bfcl
	go run ./cmd/diagnose-k -model laya -url http://localhost:8081 -dataset bfcl -nodesc
	go run ./cmd/diagnose-k -model kev-0.8b -url http://localhost:8082 -dataset clinc150
	go run ./cmd/diagnose-k -model kev-0.8b -url http://localhost:8082 -dataset bfcl
	go run ./cmd/diagnose-k -model kev-4b -url http://localhost:8083 -dataset clinc150
	go run ./cmd/diagnose-k -model kev-4b -url http://localhost:8083 -dataset bfcl
	go run ./cmd/diagnose-k -model kev-9b -url http://localhost:8084 -dataset clinc150
	go run ./cmd/diagnose-k -model kev-9b -url http://localhost:8084 -dataset bfcl

# 6. methods on the full option set: clinc150 (k=150) and bfcl (k=255), test split, 500 examples
tournament:
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method single
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method permute -r 4
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method permute -r 8
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method tournament -g 8
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method tournament -g 12
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method tournament -g 24
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method adaptive -g 12 -eps 0.05
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method adaptive -g 12 -eps 0.1
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method chain -g 12
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -method chain -g 24
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method single
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method permute -r 4
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method permute -r 8
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method tournament -g 8
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method tournament -g 12
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method tournament -g 24
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method adaptive -g 12 -eps 0.05
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method adaptive -g 12 -eps 0.1
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method chain -g 12
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method chain -g 24
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method single -nodesc
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method permute -r 8 -nodesc
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method tournament -g 8 -nodesc
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method tournament -g 24 -nodesc
	go run ./cmd/tournament -model laya -url http://localhost:8081 -dataset bfcl -k 255 -method adaptive -g 12 -eps 0.05 -nodesc
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method single
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method permute -r 4
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method permute -r 8
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method tournament -g 8
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method tournament -g 12
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method tournament -g 24
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method adaptive -g 12 -eps 0.05
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method adaptive -g 12 -eps 0.1
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method chain -g 12
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -method chain -g 24
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method single
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method permute -r 4
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method permute -r 8
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method tournament -g 8
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method tournament -g 12
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method tournament -g 24
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method adaptive -g 12 -eps 0.05
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method adaptive -g 12 -eps 0.1
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method chain -g 12
	go run ./cmd/tournament -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -method chain -g 24
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method single
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method permute -r 4
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method permute -r 8
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method tournament -g 8
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method tournament -g 12
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method tournament -g 24
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method adaptive -g 12 -eps 0.05
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method adaptive -g 12 -eps 0.1
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method chain -g 12
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -method chain -g 24
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method single
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method permute -r 4
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method permute -r 8
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method tournament -g 8
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method tournament -g 12
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method tournament -g 24
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method adaptive -g 12 -eps 0.05
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method adaptive -g 12 -eps 0.1
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method chain -g 12
	go run ./cmd/tournament -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -method chain -g 24
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method single
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method permute -r 4
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method permute -r 8
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method tournament -g 8
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method tournament -g 12
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method tournament -g 24
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method adaptive -g 12 -eps 0.05
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method adaptive -g 12 -eps 0.1
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method chain -g 12
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -method chain -g 24
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method single
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method permute -r 4
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method permute -r 8
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method tournament -g 8
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method tournament -g 12
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method tournament -g 24
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method adaptive -g 12 -eps 0.05
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method adaptive -g 12 -eps 0.1
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method chain -g 12
	go run ./cmd/tournament -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -method chain -g 24

# 7. conformal: adaptive tournament over an eps grid on calibration and test split
calibrate:
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -g 12 -split calibration
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -g 12 -split test
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -g 12 -split calibration
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -g 12 -split test
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -g 12 -split calibration
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -g 12 -split test
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -g 12 -split calibration
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -g 12 -split test
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset bfcl -k 255 -g 12 -split calibration
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset bfcl -k 255 -g 12 -split test
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -g 12 -split calibration
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -g 12 -split test
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -g 12 -split calibration
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -g 12 -split test
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -g 12 -split calibration
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -g 12 -split test
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -g 24 -split calibration
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -g 24 -split test
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -g 24 -split calibration
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -g 24 -split test
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -g 24 -split calibration
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -g 24 -split test
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -g 24 -split calibration
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -g 24 -split test
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset bfcl -k 255 -g 24 -split calibration
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset bfcl -k 255 -g 24 -split test
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -g 24 -split calibration
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -g 24 -split test
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -g 24 -split calibration
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -g 24 -split test
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -g 24 -split calibration
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -g 24 -split test
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -g 48 -split calibration
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset clinc150 -k 150 -g 48 -split test
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -g 48 -split calibration
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset clinc150 -k 150 -g 48 -split test
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -g 48 -split calibration
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset clinc150 -k 150 -g 48 -split test
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -g 48 -split calibration
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset clinc150 -k 150 -g 48 -split test
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset bfcl -k 255 -g 48 -split calibration
	go run ./cmd/calibrate -model laya -url http://localhost:8081 -dataset bfcl -k 255 -g 48 -split test
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -g 48 -split calibration
	go run ./cmd/calibrate -model kev-0.8b -url http://localhost:8082 -dataset bfcl -k 255 -g 48 -split test
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -g 48 -split calibration
	go run ./cmd/calibrate -model kev-4b -url http://localhost:8083 -dataset bfcl -k 255 -g 48 -split test
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -g 48 -split calibration
	go run ./cmd/calibrate -model kev-9b -url http://localhost:8084 -dataset bfcl -k 255 -g 48 -split test

# 8. aggregate output/ into output/eval/*.json and paper/*.gen.tex
eval:
	go run ./cmd/eval-baseline
	go run ./cmd/eval-diagnose-k
	go run ./cmd/eval-tournament
	go run ./cmd/eval-calibrate

test:
	go vet ./... && go test ./...

clean:
	rm -rf output paper/*.gen.tex
