# Systemonebench

## Run all models

```sh
mkdir -p models
curl -L -o models/Laya-BF16.gguf https://huggingface.co/ggml-org/Laya-GGUF/resolve/main/Laya-BF16.gguf
curl -L -o models/Kev-0.8B-BF16.gguf https://huggingface.co/ggml-org/Kev-0.8B-GGUF/resolve/main/Kev-0.8B-BF16.gguf
curl -L -o models/Kev-4B-BF16.gguf https://huggingface.co/ggml-org/Kev-4B-GGUF/resolve/main/Kev-4B-BF16.gguf
curl -L -o models/Kev-9B-BF16.gguf https://huggingface.co/ggml-org/Kev-9B-GGUF/resolve/main/Kev-9B-BF16.gguf
curl -L -o models/Qwen2.5-7B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
docker compose up -d --build
curl localhost:8081/health   # laya
curl localhost:8082/health   # kev-0.8b
curl localhost:8083/health   # kev-4b
curl localhost:8084/health   # kev-9b
curl localhost:8090/health   # qwen-7b (LLM baseline)
```

## Datasets

```sh
python3 scripts/fetch_data.py   # -> data/<name>/{options.json,calibration.jsonl,test.jsonl}, 50/50 split, seed 0
```

| name | source | rows | options |
|---|---|---|---|
| emotion | dair-ai/emotion test | 2000 | 6 |
| xnli | facebook/xnli en test | 5010 | 3 |
| massive | mteb/amazon_massive_intent en test | 2974 | 59 |
| banking77 | mteb/banking77 test | 3076 | 77 |
| clinc150 | DeepPavlov/clinc150 test (out-of-scope rows skipped) | 4500 | 150 |
| bfcl | BFCL v3 simple + live_simple (tool pool as options) | 658 | 453 |

## Models

* [Laya GGUF](https://huggingface.co/ggml-org/Laya-GGUF)
* [Laya](https://huggingface.co/convaiinnovations/laya)
* [Kev-0.8B GGUF](https://huggingface.co/ggml-org/Kev-0.8B-GGUF)
* [Kev-4B GGUF](https://huggingface.co/ggml-org/Kev-4B-GGUF)
* [Kev-9B GGUF](https://huggingface.co/ggml-org/Kev-9B-GGUF)
* [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b)
* [llama.cpp /v1/systemone support](https://github.com/ggml-org/llama.cpp/pull/29818)

## Datasets

* [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive)
* [XNLI](https://huggingface.co/datasets/facebook/xnli)
* [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)
* [DAIR Emotion](https://huggingface.co/datasets/dair-ai/emotion)
* [Banking77](https://huggingface.co/datasets/PolyAI/banking77)
* [SST-5](https://huggingface.co/datasets/SetFit/sst5)
* [Prompt Injections](https://huggingface.co/datasets/deepset/prompt-injections)
* [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat)
* [DBPedia Classes](https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes)
* [Web of Science](https://huggingface.co/datasets/river-martin/web-of-science-with-label-texts)

## Bibliography

* [Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents](https://arxiv.org/abs/2609.28940)
* [Laya – GitHub repository](https://github.com/NandhaKishorM/laya)
* [Laya – Hugging Face model card](https://huggingface.co/Emerald7664/laya)
* [Laya – benchmark report](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)
* [Laya – PyPI package](https://pypi.org/project/laya/)
* [I built non-autoregressive decision models a year ago, then a frontier lab called it a…](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)
* [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks)
* [decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)
* [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
* [Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)](https://arxiv.org/abs/1911.02116)
* [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)](https://arxiv.org/abs/2412.13663)
* [mmBERT: A Modern Multilingual Encoder with Annealed Language Learning](https://arxiv.org/abs/2509.06888)
* [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
* [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
* [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
* [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
* [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281)
* [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676)
* [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/abs/2012.15723)
* [Efficient Few-Shot Learning Without Prompts (SetFit)](https://arxiv.org/abs/2209.11055)
* [Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach](https://arxiv.org/abs/1909.00161)
* [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE)](https://doi.org/10.1007/BF00992696)
* [Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning](https://www.jmlr.org/papers/v5/greensmith04a.html)
* [Learning to Predict by the Methods of Temporal Differences](https://doi.org/10.1007/BF00115009)
* [Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300)
* [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
* [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (RLOO)](https://arxiv.org/abs/2402.14740)
* [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
* [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
* [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
* [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
* [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
* [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437)
* [Verification of Forecasts Expressed in Terms of Probability (Brier score)](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2)
* [A Scoring System for Probability Forecasts of Ranked Categories (RPS)](https://doi.org/10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2)
* [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
* [Measuring Calibration in Deep Learning](https://arxiv.org/abs/1904.01685)
* [Verified Uncertainty Calibration](https://arxiv.org/abs/1909.10155)
* [Calibrating Deep Neural Networks using Focal Loss](https://arxiv.org/abs/2002.09437)
* [Calibration of Pre-trained Transformers](https://arxiv.org/abs/2003.07892)
* [Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://arxiv.org/abs/1906.02530)
* [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474)
* [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511)
* [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)
* [Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334)
* [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://arxiv.org/abs/2305.14975)
* [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://arxiv.org/abs/2306.13063)
* [Reducing Conversational Agents' Overconfidence through Linguistic Calibration](https://arxiv.org/abs/2012.14983)
* [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
* [Consistent Estimators for Learning to Defer to an Expert](https://arxiv.org/abs/2006.01862)
* [From System 1 to System 2: A Survey of Reasoning Large Language Models](https://arxiv.org/abs/2502.17419)
* [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
* [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
* [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176)
* [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)
* [PentestGPT: An LLM-empowered Automatic Penetration Testing Tool](https://arxiv.org/abs/2308.06782)
* [MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages](https://arxiv.org/abs/2204.08582)
* [XNLI: Evaluating Cross-lingual Sentence Representations](https://arxiv.org/abs/1809.05053)
* [Character-level Convolutional Networks for Text Classification (AG News)](https://arxiv.org/abs/1509.01626)
* [CARER: Contextualized Affect Representations for Emotion Recognition (DAIR Emotion)](https://aclanthology.org/D18-1404/)
* [Efficient Intent Detection with Dual Sentence Encoders (Banking77)](https://arxiv.org/abs/2003.04807)
* [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank (SST-5)](https://aclanthology.org/D13-1170/)
