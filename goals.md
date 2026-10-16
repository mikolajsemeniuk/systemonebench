# Goal

Modele System 1 (Laya, Kev) przy dużej liczbie opcji: diagnoza, kiedy pojedyncze pytanie zawodzi, i metoda pytania (turniej), która to naprawia z gwarancją i znanym kosztem.

## Problem

* Przy wielu opcjach (59+) Laya odpowiada źle w połowie przypadków i jest pewna na 100% (acc 0.46, ECE 0.46).
* Kolejność opcji zmienia decyzję: odwrócenie listy zmienia 54% odpowiedzi Laya i 26% Kev-4B (MASSIVE, 59 opcji).
* Laya dzieli budżet 192 tokenów między wszystkie opcje: przy 50 opcjach ~4 tokeny na opcję.
* Realne zastosowania mają dużo opcji: Banking77 (77), CLINC-150 (150), agenty z 30–50 narzędziami, API Jev przyjmuje do 255.
* Literatura (2610.02267) mierzy problem i radzi „trzymaj ≤ 10 opcji”. Nikt nie testuje naprawy.

## Etap 1: diagnoza

* Accuracy, ECE, flip rate vs liczba opcji (k = 3 … 150), długość opcji, opisy tak/nie, kolejność.
* Modele: Laya, Kev-0.8B / 4B / 9B. Datasety: MASSIVE, Banking77, CLINC-150, Emotion, XNLI, wybór narzędzi (BFCL).
* Wynik: reguły promptu per model (ile opcji, jak długie).

## Etap 2: metoda

```
77 opcji ──► losowy podział na grupy po G
                │
      wywołanie per grupa, przechodzą opcje z p ≥ ε
                │
      powtarzaj aż ≤ G opcji ──► finał (1 wywołanie)
                │
      R losowych podziałów, średnia prawdopodobieństw
```

* Uśrednianie permutacji (R): usuwa zależność od kolejności.
* Turniej (G): każde wywołanie ma mało opcji, pełny budżet tokenów na opcję.
* Próg ε: łatwe pytania kończą się po 1–2 wywołaniach, trudne dostają więcej.
* ε dobierane conformal na danych kalibracyjnych: poprawna opcja dochodzi do finału z p ≥ 1 − α.
* Wariant bez etykiet: zgodność z drogim turniejem zamiast poprawności.

## Pilot (n = 300, MASSIVE, 59 opcji, GX10)

| model | wariant | acc | ECE | stabilność |
|---|---|---|---|---|
| Laya | 1 wywołanie | 0.457 | 0.461 | flip 54% |
| Laya | turniej G=8 (9 wywołań) | 0.573 | 0.155 | gold w finale 82% |
| Kev-4B | 1 wywołanie | 0.710 | 0.242 | flip 26% |
| Kev-4B | 8 permutacji | 0.750 | 0.287 | flip 4% |
| Kev-4B | turniej G=8 | 0.697 | 0.137 | gold w finale 90% |

## Nowość

* Pierwsza metoda inferencyjna dla modeli System 1 obsługująca dowolnie duży zbiór opcji, bez treningu, po stronie klienta.
* Gwarancja conformal, że poprawna opcja nie odpadnie, z pokrętłem α.
* Koszt adaptacyjny do trudności pytania.
* Działa z każdym modelem /v1/systemone (Laya, Kev, Jev, Microsoft Decision-1).

## Struktura artykułu

1. Diagnoza: kiedy pojedyncze pytanie zawodzi (k, długość, kolejność), Laya vs Kev.
2. Metoda: permutacje, turniej, ε, gwarancja.
3. Wyniki: siatka G × R × α, front Pareto koszt vs accuracy, weryfikacja gwarancji.
4. Transfer: G i R między datasetami, ε per zadanie.
5. Odniesienie: LLM (Qwen 7B) accuracy i latencja.

## Metryki

* Jakość: accuracy, macro-F1, ECE, Brier.
* Stabilność: flip rate (odwrócenie, między losowaniami).
* Gwarancja: survival rate golda vs 1 − α.
* Koszt: wywołania/decyzję, latencja p50/p95, J/decyzję.

## Infrastruktura

* Dockerfile: llama.cpp CUDA arm64 (działa, b11559).
* `docker-compose.yaml`: `llama-server` per model.
* Go: datasety, klient `/v1/systemone`, turniej, metryki, koszt.
* Sprzęt: ASUS Ascent GX10 (GB10, aarch64, 128 GB).

## Ryzyka

* Kev-9B może radzić sobie z 150 opcjami w jednym wywołaniu: wtedy metoda głównie dla małych enkoderów.
* ε nie przenosi się między datasetami: kalibracja per zadanie (minuta, kilkaset przykładów).
* Port llama.cpp vs oryginał PyTorch/PEFT niezweryfikowany.
* Literatura z października 2026 szybko rośnie (AnyJev robi permutacje dla LLM): sprawdzić przed pisaniem.
