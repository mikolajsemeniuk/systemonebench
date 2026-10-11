## Ideas taken:

* Dos Santos (2609.28940): Laya/Jev jako warstwa decyzyjna System 1 w agencie pentestowym; studium przypadku z 1 przebiegiem na wariant, ECE Laya porównane wyłącznie na podstawie publikowanych danych, kalibracja na zadaniach pentestowych niezmierzona.
* Repo Laya: SDK, CLI, HTTP `laya-serve` z `/v1/systemone` (zgodny z Jev), Docker/Compose, ONNX, router między checkpointami (EN/multilingual).
* Karta HF Laya: ModernBERT-large + głowica decyzyjna (421M), scoring opcji na `[MASK]`, RLCD = REINFORCE + baseline grupowy + proper scoring rules.
* BENCHMARKS.md Laya: wszystko mierzone przez autorów, 1 seed, dane częściowo z treningu; ECE 0.466→0.081 po temperaturze; Khmer 0.000 acc przy 0.952 conf; flip rate od kolejności opcji do 0.23; GB10 (aarch64) na współdzielonym GPU z niewyjaśnionym narzutem ~93 ms.
* PyPI Laya: dystrybucja pakietu Python.
* Dev.to: narracja autora o modelach decyzyjnych bez autoregresji, bez ewaluacji.
* jev-benchmarks: zewnętrzne benchmarki Jev, nie Laya.
* decision-model-benchmark: zewnętrzne porównanie modeli decyzyjnych, bez kontroli sprzętu i kalibracji.
* Transformer: mechanizm self-attention, baza architektury.
* BERT: dwukierunkowy enkoder + MLM, wzorzec klasyfikacji przez enkoder.
* RoBERTa: lepszy recipe pretrainingu enkodera.
* ELECTRA: wydajny pretraining enkodera przez dyskryminację tokenów.
* DeBERTa: rozdzielona uwaga treść/pozycja w enkoderze.
* XLM-R: wielojęzyczny enkoder, baseline dla transferu cross-lingual.
* ModernBERT: nowoczesny enkoder (RoPE, GeGLU, długi kontekst) – backbone Laya EN.
* mmBERT: wielojęzyczny ModernBERT (1800+ języków) – backbone Laya multilingual.
* RoFormer: rotary position embedding używane w ModernBERT.
* GLU Variants: GeGLU w FFN ModernBERT.
* FlashAttention: szybka uwaga, wpływa na latencję inferencji.
* Sentence-BERT: enkoder jako szybka alternatywa dla cross-attention LLM.
* NAT: dekodowanie nieautoregresyjne – idea „jeden forward pass”.
* PET: klasyfikacja przez cloze/`[MASK]` – bezpośredni poprzednik scoringu opcji Laya.
* LM-BFF: prompt + few-shot dla enkoderów.
* SetFit: few-shot klasyfikacja bez promptów, tani baseline.
* Zero-shot NLI (Yin): klasyfikacja zero-shot przez entailment – baseline dla typowanych pytań.
* REINFORCE: gradient polityki, rdzeń RLCD.
* Greensmith: redukcja wariancji przez baseline – baseline grupowy w RLCD.
* TD(λ) (Sutton 1988): kredyt w wieloturowym RLCD.
* Sutton & Barto: podstawy RL.
* PPO: standardowe RL dla LLM, kontekst dla GRPO.
* DeepSeekMath: GRPO – baseline grupowy, użyty w RLCD.
* DeepSeek-R1: RL z nagrodą weryfikowalną na dużą skalę.
* RLOO: REINFORCE zamiast PPO wystarcza do RLHF.
* Christiano: RL z preferencji ludzkich.
* InstructGPT: RLHF dla LLM.
* Constitutional AI: RLAIF – feedback od AI.
* RLAIF vs RLHF: RLAIF dorównuje RLHF.
* DPO: alternatywa bez RL dla preferencji.
* Gneiting & Raftery: teoria proper scoring rules – nagroda RLCD.
* Brier: Brier score.
* Epstein: RPS dla kategorii porządkowych – nagroda dla pytań `score`.
* Guo et al.: przekalibrowanie sieci, temperature scaling, ECE – użyte przez Laya.
* Nixon et al.: wady ECE, adaptive/class-wise ECE.
* Kumar et al.: ECE z binningiem jest obciążone, estymator z debiasingiem.
* Mukhoti et al.: focal loss poprawia kalibrację.
* Desai & Durrett: kalibracja pretrenowanych transformerów in/out-of-domain.
* Ovadia et al.: kalibracja degraduje pod przesunięciem rozkładu.
* Deep Ensembles: ensemble daje lepszą niepewność.
* Conformal prediction (Angelopoulos): zbiory predykcji z gwarancją pokrycia.
* Kadavath et al.: LLM częściowo wiedzą, czego nie wiedzą (P(True)).
* Lin et al.: werbalizowana niepewność.
* Tian et al.: werbalizowana pewność RLHF-LLM lepiej skalibrowana niż logity.
* Xiong et al.: ewaluacja elicytacji pewności LLM, nadmierna pewność.
* Mielke et al.: lingwistyczna kalibracja agentów.
* Geifman & El-Yaniv: selective classification, risk–coverage.
* Mozannar & Sontag: learning to defer do eksperta.
* System 1 → System 2 survey: podział szybkie/wolne myślenie w LLM.
* ReAct: agent rozumowanie + akcje – miejsce na decyzje System 1.
* Toolformer: LLM samo decyduje o narzędziach.
* FrugalGPT: kaskada modeli redukuje koszt.
* RouteLLM: router tani/drogi model z preferencji.
* PentestGPT: LLM w pentestach – domena 2609.28940.
* MASSIVE: wielojęzyczne intencje, 51 języków – benchmark Laya.
* XNLI: wielojęzyczne NLI – benchmark Laya.
* AG News: klasyfikacja tematów – benchmark Laya.
* CARER/DAIR Emotion: emocje 6 klas – benchmark Laya.
* Banking77: 77 intencji – Laya słaba przez budżet tokenów opcji.
* SST-5: sentyment porządkowy – najsłabszy prymityw Laya (`score`).

## Potential ideas:

* **(rekomendowane, rdzeń artykułu)** Pierwsza niezależna, reprodukowalna ewaluacja Laya: harness w Go, Laya przez HTTP `/v1/systemone` w `docker-compose.yaml` z własnego Dockerfile, ASUS Ascent GX10 (GB10, aarch64), wiele seedów, przedziały ufności z bootstrapu, tylko dane spoza treningu – braki wprost przyznane w BENCHMARKS.md.
* Pełny audyt kalibracji w Go (`metrics.go`): accuracy, macro-F1, NLL, Brier, RPS (pytania `score`), ECE/MCE/adaptive ECE/debiased ECE, reliability diagrams, AURC i accuracy@coverage; dotąd jest tylko jedna kolumna ECE z jednym seedem.
* Conformal prediction nad Laya: zbiory predykcji z gwarancją pokrycia zamiast kruchych progów `min_confidence` (autorzy piszą, że pewność „nie ustala poprawności”) – nikt tego nie zrobił dla modeli System 1.
* Laya na GB10 izolowany: rozbicie narzutu ~93 ms (tokenizacja, transfer, CUDA graphs, HTTP), PyTorch vs ONNX vs TensorRT na aarch64, latencja p50/p95/p99 vs batch, throughput pod współbieżnym obciążeniem z Go.
* Energia na decyzję (J/decyzję) na GB10 z odczytu mocy GPU – metryka nieobecna w żadnym źródle, dobrze się sprzedaje („green AI”, edge).
* System 1 vs System 2 na tym samym sprzęcie: Laya vs mały lokalny LLM (np. przez vLLM w tym samym compose) na identycznych pytaniach typowanych – front Pareto accuracy/ECE/latencja/energia.
* Kaskada Laya → LLM (FrugalGPT/RouteLLM dla decyzji): Laya odpowiada, gdy pewna, eskaluje gdy nie; mierzymy oszczędność kosztu przy stałej dokładności – mocny wynik praktyczny.
* Pewność pod przesunięciem rozkładu: przypadek Khmer (0.000 acc, 0.952 conf) zbadany systematycznie na MASSIVE/XNLI per język + prosty detektor OOD (pismo/język, entropia) jako bramka.
* Wrażliwość na kolejność i liczbę opcji: flip rate przy pełnych permutacjach, dryf pewności z liczbą opcji (autorzy to przyznają, nie mierzą), sequential vs parallel layout.
* Odporność adwersarialna i prompt injection dla presetu guardrail: held-out 0.698 na deepset, brak testów ataków na samą bramkę decyzyjną.
* Wariancja między seedami RLCD: fine-tuning kilka razy na GB10 i pomiar rozrzutu accuracy/ECE – autorzy trenowali 1 seed na wariant.
* Ewaluacja na decyzjach pentestowych (luka z 2609.28940: „kalibracja na zadaniach pentestowych niezmierzona”) – zbiór decyzji DP1–DP4 i pomiar kalibracji Laya.
