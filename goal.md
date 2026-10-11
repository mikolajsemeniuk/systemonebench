# Goal

Kaskada Laya (skwantyzowana) → Kev-4B: jakość ≈ Kev-4B, koszt ≈ Laya, z gwarancją zgodności bez etykiet.

## Problem

* Kev-4B (Qwen3.5-4B + LoRA, ~10× większy) jest prawdopodobnie lepszy od Laya (421M), ale droższy dla każdej decyzji.
* Laya jest kwantyzowana przez społeczność, ale nikt nie zmierzył wpływu kwantyzacji na trafność i kalibrację.
* Nikt nie zaproponował kaskady modeli decyzyjnych System 1 z gwarancją zgodności.

## Etap 1: kwantyzacja Laya

* Laya BF16 vs Q8_0, Q6_K, Q4_K_M, Q4_0, Q2_K (wszystkie z tego samego BF16 przez `llama-quantize`).
* Pomiar: trafność, kalibracja, flip rate vs BF16, flip rate vs margines, latencja, energia.
* Wynik: najtańszy poziom kwantyzacji, który nie psuje jakości (sweet spot kwantyzacji).

## Etap 2: kaskada Laya-Qx → Kev-4B

```
pytanie ──► Laya-Qx (tania)
               │
     margines = p(top1) − p(top2)
               │
      ┌────────┴────────┐
 margines ≥ τ        margines < τ
      │                 │
 odpowiedź Laya     Kev-4B (drogi)
```

* Laya-Qx = sweet spot z etapu 1.
* Flip = Laya-Qx ≠ Kev-4B.
* `τ` dobierane conformal na danych bez etykiet dla tej pary: zgodność z Kev-4B ≥ 1 − α (np. 99%).
* Gwarancja dotyczy zgodności z Kev-4B, nie poprawności.

## Nowość

* Kaskada modeli decyzyjnych System 1 z gwarancją zgodności z dużym modelem, bez etykiet.
* Tani poziom dodatkowo skwantyzowany – oszczędność na dwóch osiach: precyzja i rozmiar.

## Struktura artykułu

1. Laya vs Kev-4B na tych samych danych – czy Kev-4B jest wyraźnie lepszy (warunek sensu kaskady).
2. Etap 1: kwantyzacja Laya – sweet spot.
3. Etap 2: kaskada Laya-Qx → Kev-4B, dobór `τ`.
4. Wynik: front Pareto – trafność ≈ Kev-4B, koszt ≈ Laya.
5. Plan B (jeśli kwantyzacja nie szkodzi): ten sam wniosek wzmacnia kaskadę – Laya-Q2/Q4 jako najtańszy poziom.

## Modele (llama.cpp `/v1/systemone`)

* [Laya GGUF](https://huggingface.co/ggml-org/Laya-GGUF) – ModernBERT-large, 421M, Apache-2.0.
* [Kev-4B GGUF](https://huggingface.co/ggml-org/Kev-4B-GGUF) – Qwen3.5-4B-Base + LoRA, Apache-2.0, gotowe BF16 / Q8_0 / Q4_K_M.

## Datasety

* Bez przecieku do Kev-4B: MASSIVE, XNLI, DAIR Emotion.
* Kev-4B trenowany na Banking77, AG News, SST-5, DBpedia – wyniki tam zawyżone, tylko pomocniczo.

## Metryki

* Jakość: accuracy, macro-F1, ECE, Brier, NLL.
* Zgodność: flip rate, flip rate vs margines, empiryczna zgodność kaskady vs 1 − α.
* Koszt: latencja p50/p95/p99, throughput, J/decyzję, % eskalacji do Kev-4B.

## Infrastruktura

* Jeden Dockerfile: llama.cpp CUDA arm64.
* `docker-compose.yaml`: `llama-server` per model/kwantyzacja (Laya-Qx, Laya-BF16, Kev-4B).
* Go: wczytanie datasetów z HF, klient `/v1/systemone`, zapis pełnych prawdopodobieństw, metryki offline, kaskada, pomiar latencji i energii.
* Sprzęt: ASUS Ascent GX10 (GB10, aarch64, Blackwell, 128 GB).

## Do weryfikacji przed startem

* llama.cpp CUDA buduje się i działa na GB10.
* llama.cpp BF16 zgodny z oryginałem (Laya PyTorch, Kev-4B PEFT).
* Kev-4B wyraźnie lepszy od Laya na MASSIVE/XNLI/Emotion (jeśli nie – kaskada bez sensu).
* Flipy skupione przy małym marginesie.
* Przegląd literatury: kaskady z gwarancją (conformal), speculative decoding.
