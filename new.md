# New findings (pilot, 2026-10-16)

Wyniki pilota na GX10, ustalone w jednej sesji. Mała próba: n = 300 na dataset, jedno sformułowanie pytania, seed 0. To sygnał kierunkowy, a nie wyniki do artykułu.

## Środowisko i infrastruktura

- Maszyna: ASUS Ascent GX10, aarch64, NVIDIA GB10 (compute capability 12.1), sterownik 580.178.04, Docker 29.6.2 z CDI (`--gpus all` działa). Pamięć 121 GB, współdzielona między CPU i GPU.
- Obraz `systemonebench` (Dockerfile z repo, llama.cpp `b11559`, `CUDA_ARCH=121`) **buduje się i działa**. Oba modele działają na GPU i odpowiadają na `/v1/systemone`.
- Pułapka: binarki są statycznie zlinkowane z CUDA. **Bez `--gpus all` nie startują** (`libcuda.so.1: cannot open shared object file`), dotyczy to też `llama-quantize` i trybu CPU. Tryb CPU: `--gpus all ... --device none`. Przykład w README jest już poprawiony.
- Pułapka: kiedy na GPU działał inny workload (vLLM, ok. 64 GB), losowo pojawiało się `CUDA error: out of memory` przy `cudaSetDevice`/kwantyzacji. Wystarczało ponowić. Pomiary latencji Laya i Kev-4B zrobiono **przy działającym vLLM** (szum). Pomiary LLM zrobiono po jego wyłączeniu.
- Pułapka: `llama-quantize` w kontenerze zapisuje pliki jako root. Trzeba używać `--user $(id -u):$(id -g)`.
- Kev-4B domyślnie bierze kontekst 262k tokenów (ok. 17 GB). Z `-c 8192` zajmuje dużo mniej.
- `docker run` dla serwera: `docker run -d --gpus all -p 8081:8080 -v $PWD/models:/models:ro systemonebench -m /models/Laya-BF16.gguf -ngl 99`
- Kwantyzacja: `docker run --rm --gpus all --user $(id -u):$(id -g) -v $PWD/models:/models --entrypoint /app/llama-quantize systemonebench /models/Laya-BF16.gguf /models/Laya-Q4_K_M.gguf Q4_K_M`

### Modele w `models/` (w .gitignore)
- Laya: BF16 (pobrany z ggml-org/Laya-GGUF), F16, Q8_0, Q6_K, Q4_K_M, Q4_0 i Q2_K (zrobione `llama-quantize` z BF16). Rozmiary: 803 → 427 / 347 / 272 / 239 / 186 MiB. Q2_K ma efektywnie 3.71 BPW, bo część tensorów jest w wyższej precyzji.
- Kev-4B: BF16, Q8_0, Q4_K_M (pobrane z ggml-org/Kev-4B-GGUF).
- llama.cpp `b11559` wspiera natywnie typ decyzyjny `kev` (`tools/server/server-decision.cpp`) oraz `laya`. Laya: max 255 opcji, prompt musi zmieścić się w `--ubatch-size`, długie opcje są przycinane do budżetu tokenów.

### API `/v1/systemone` (z tools/server/README.md)
Żądanie: `{"state": str|obj, "questions": {"id": {"type": "choice"|"score"|"noul", "instructions": "...", "criteria": {...}}}}`.
Odpowiedź `choice`: `choice`, `probabilities` (suma 1, skalowane temperaturą z pliku modelu), `confidence`.

### Dane
- HF datasets-server `/rows` (stdlib, paginacja po 100, retry z backoffem, bo API losowo zwraca błędy).
- MASSIVE: oryginał `AmazonScience/massive` używa skryptu i nie działa w datasets-server. Używamy mirrora **`mteb/amazon_massive_intent`** (config `en`, test 2974, 59 intencji w teście).
- XNLI `facebook/xnli` config `en` (test 5010). DAIR Emotion `dair-ai/emotion` config `split` (test 2000).
- Próbka: 300 przykładów na dataset, `random.Random(0).shuffle`.
- Pytania: Emotion „Which emotion does the author of this text express?”, opcje bez opisów. XNLI: state = {premise, hypothesis}, opcje z opisami. MASSIVE: surowe etykiety (`alarm_set`) bez opisów, co może krzywdzić Laya.

## Wyniki

### 1. Laya BF16 vs Kev-4B BF16 (accuracy, paired bootstrap 95% CI)
| dataset | Laya | Kev-4B | Δ |
|---|---|---|---|
| Emotion | 0.573 | 0.563 | −0.010 [−0.063, +0.043] |
| XNLI | 0.870 | 0.877 | +0.007 [−0.030, +0.043] |
| MASSIVE | 0.457 | 0.710 | +0.253 [+0.183, +0.320] |

Kev-4B jest wyraźnie lepszy tylko na MASSIVE, a tam możliwy jest efekt surowych etykiet. Flip rate Laya≠Kev: 34% / 12% / 56%. Flipy maleją z marginesem Laya, ale na MASSIVE Laya jest skrajnie nadmiernie pewna (40% przykładów ma margines 1.000). Kaskada Laya → Kev przy zgodności 99% wymaga 48–95% eskalacji.

### 2. Kwantyzacja Laya (względem Laya BF16)
| dataset | poziom | acc | ECE | flipy/300 | flipy w dolnym kwintylu marginesu BF16 |
|---|---|---|---|---|---|
| Emotion | Q8_0 | 0.573 | 0.323 | 3 | 3 |
| Emotion | Q4_K_M | 0.583 | 0.309 | 21 | 20 |
| Emotion | Q2_K | 0.540 | 0.257 | 67 | 30 |
| XNLI | Q8_0 | 0.870 | 0.044 | 0 | 0 |
| XNLI | Q4_K_M | 0.883 | 0.029 | 8 | 7 |
| XNLI | Q2_K | 0.863 | 0.044 | 29 | 21 |
| MASSIVE | Q8_0 | 0.460 | 0.456 | 15 | 15 |
| MASSIVE | Q4_K_M | 0.410 | 0.484 | 61 | 31 |
| MASSIVE | Q2_K | 0.403 | 0.474 | 140 | 48 |

- Accuracy i ECE nie wykrywają szkód: Q4_K_M na Emotion ma +1 pp accuracy przy 7% zmienionych decyzji, Q2_K ma *lepsze* ECE przy 22% flipów.
- Kontrolowany eksperyment z liczbą opcji (te same przykłady MASSIVE, opcje = złota + losowe dystraktory). Flip rate:

| k opcji | Q8_0 | Q4_K_M | Q2_K |
|---|---|---|---|
| 3 | 0.010 | 0.027 | 0.097 |
| 6 | 0.010 | 0.050 | 0.130 |
| 12 | 0.013 | 0.047 | 0.240 |
| 24 | 0.010 | 0.093 | 0.290 |
| 59 | 0.050 | 0.203 | 0.467 |

### 3. Latencja (GB10, jedno zapytanie, emotion, p50)
| model | GPU p50 | GPU throughput (c=8) | CPU p50 |
|---|---|---|---|
| Laya BF16 | 7.7 ms | 401 req/s | 856 ms (wolne kernele BF16 na CPU ARM) |
| Laya F16 | — | — | 82 ms |
| Laya Q8_0 | 7.1 ms | 353 | 83 ms |
| Laya Q4_K_M | 6.7 ms | 357 | 85 ms |
| Laya Q2_K | 6.4 ms | 361 | 101 ms |
| Kev-4B BF16 | 62.1 ms | 20 req/s | — |
| Kev-4B Q8_0 | 49.8 ms | 26 | — |
| Kev-4B Q4_K_M | 42.3 ms | 32 | — |
| Qwen2.5-7B-Instruct Q4_K_M (Ollama, JSON enum, c=4) | 546–1138 ms | — | — |

- **Kwantyzacja Laya nie daje zysku latencji ani na GPU, ani na CPU.** Pozorne 10× na CPU było artefaktem BF16; F16 jest tak samo szybki jak Q4. Daje tylko mniejszy plik. Prawdopodobna przyczyna: jeden przebieg prefill na krótkim promptcie, a przy 421M parametrów wąskim gardłem nie jest przesył wag z pamięci.
- **Kwantyzacja Kev-4B daje 1.47×** przy 1.3–5% flipów względem Kev BF16, prawie wszystkie w dolnym kwintylu marginesu.
- Strażnik conformal (Kev-Q4 → Kev-BF16 przy niskim marginesie) przy α = 1%: średnio 45 ms (Emotion, XNLI) lub 57 ms (MASSIVE) zamiast 62 ms.

### 4. Conformal risk control bez etykiet — działa
Strata: `1[margines ≥ τ i tani ≠ referencja]`. Wybór: najmniejsze τ takie, że `(n·R̂(τ)+1)/(n+1) ≤ α`. W 500 losowych podziałach 50/50 kalibracja/test **średnia niezgodność na teście była ≤ α we wszystkich konfiguracjach** (gwarancja jest w oczekiwaniu, więc P(pojedynczy podział ≤ α) wynosi ok. 0.6–1.0).

### 5. Obecny kierunek: offload wywołań LLM do modeli System 1 (praktyczna obniżka kosztów)
Pomysł: firma klasyfikuje dziś LLM-em (routing, intencje, guardrail, wybór narzędzia). Na próbce ruchu z logów (same odpowiedzi LLM, bez etykiet) kalibruje τ. Model System 1 odpowiada, gdy margines ≥ τ, w pozostałych przypadkach pytanie idzie do LLM. Gwarancja: zgodność z obecnym LLM-em ≥ 1 − α. Referencyjny LLM w pilocie: `qwen2.5:7b-instruct-q4_K_M` w Ollamie (accuracy: Emotion 0.527, XNLI 0.850, MASSIVE 0.767).

Kolumny: offload = odsetek zapytań obsłużonych przez model System 1; zgodność = zgodność z LLM na teście; acc kaskady / acc LLM = accuracy kaskady i samego LLM; oszczędność = oszczędzony czas LLM.

| dataset | tani model | α | offload | zgodność | acc kaskady | acc LLM | oszczędność |
|---|---|---|---|---|---|---|---|
| Emotion | Laya | 0.05 | 39% | 0.955 | 0.543 | 0.526 | 38% |
| Emotion | Kev-Q4 | 0.05 | 52% | 0.957 | 0.543 | 0.526 | 45% |
| XNLI | Laya | 0.10 | 89% | 0.901 | 0.887 | 0.849 | 88% |
| XNLI | Laya | 0.05 | 71% | 0.954 | 0.881 | 0.850 | 70% |
| XNLI | Laya | 0.02 | 47% | 0.980 | 0.869 | 0.849 | 45% |
| MASSIVE | Laya | 0.05 | 20% | 0.957 | 0.771 | 0.767 | 19% |
| MASSIVE | Kev-Q4 | 0.05 | 47% | 0.955 | 0.800 | 0.767 | 43% |
| MASSIVE | Kev-Q4 | 0.02 | 28% | 0.982 | 0.787 | 0.769 | 24% |

- Gwarancja trzyma się wszędzie, a accuracy kaskady jest ≥ accuracy samego LLM (w każdym wierszu).
- Opłacalność silnie zależy od zadania: od 20% do 89% offloadu przy α = 5–10%.
- Na zadaniach z wieloma opcjami (MASSIVE) Kev-Q4 jest dużo lepszym tanim modelem niż Laya.

## Literatura (z agenta; część zweryfikowana tylko po fragmentach wyników wyszukiwania, przed cytowaniem przeczytać)
- arXiv 2610.00346 (Rafe i Das, 2026): benchmark modeli System 1 (Laya, Kev, Jev), kaskady z bramką top-prob do Gemma/Jev, Learn-then-Test, ale **z etykietami**, bez kwantyzacji. Najbliższa praca.
- arXiv 2510.17543 (CAb, Huang i in.): kaskada edge → cloud z conformal gwarancją **bez etykiet**, względem modelu chmurowego, na zbiorach predykcji (FDR), CIFAR/TeleQnA.
- arXiv 2607.25018 (Conformal Cascade): gwarancja poprawności, z etykietami.
- arXiv 2407.18370 (Trust or Escalate, ICLR 2025): gwarancja zgodności z ludźmi.
- arXiv 2511.07396 (C3PO): kaskada bez etykiet, gwarancja na koszcie.
- arXiv 2405.19261 (speculative cascades), 2407.02348, 2402.04513: kaskady bez gwarancji.
- arXiv 2407.09141 (Dutta i in., „Accuracy is Not All You Need”): metryka flips dla skwantyzowanych LLM; brak prac o kwantyzacji GGUF enkoderów decyzyjnych.
- arXiv 2609.33401, 2610.11392 (TypedBench), 2609.28940: inne prace o modelach System 1.

## Otwarte ryzyka / do zrobienia
1. Port llama.cpp BF16 vs oryginał w PyTorch (Laya) i PEFT (Kev): **niezweryfikowane**. Obraz `vllm/vllm-openai:latest` ma torch 2.11 i transformers 5.13, więc nadaje się na referencję.
2. MASSIVE z opisami opcji zamiast surowych etykiet.
3. Referencyjny LLM jest słaby (7B). Do praktycznej tezy potrzebny jest realistyczny „obecny system” (np. większy lokalny model albo API) oraz koszt w $ (tokeny × cennik). Pilot: LLM ok. 91–351 tokenów wejściowych i ok. 8 wyjściowych na decyzję.
4. Więcej datasetów praktycznych (Banking77, guardrail: deepset/prompt-injections, lmsys/toxic-chat). Metryka zgodności nie wymaga etykiet, więc przeciek danych treningowych Kev ma mniejsze znaczenie.
5. Pomiary latencji i energii na czystym GPU, bez innych workloadów.
6. Dryf rozkładu: czy τ skalibrowane na starym ruchu trzyma się na nowym (cotygodniowa rekalibracja?).

## Pliki pilota (scratchpad, tymczasowe; skopiować do repo, jeśli mają zostać)
`/tmp/claude-1000/-home-mikolajsemeniuk-Projects-systemonebench/5ad2b58d-219f-4d4d-822e-48933286a2b7/scratchpad/`
- `check/fetch_data.py`: pobranie i próbkowanie danych.
- `check/run_eval.py`: klient `/v1/systemone`, zapis pełnych prawdopodobieństw.
- `check/llm_eval.py`: baseline LLM przez Ollamę.
- `check/analyze.py`: Laya vs Kev.
- `check/quant_analyze.py`: kwantyzacja względem BF16.
- `check/hyp_tests.py`: liczba opcji i conformal.
- `check/offload.py`: offload LLM.
- `check/latency.py`: latencja.
- `data/`, `data_k*/`: próbki. `out/`, `out_k*/`: predykcje (jsonl per model.dataset).
