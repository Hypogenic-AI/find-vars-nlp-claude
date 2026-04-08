# Research Code Repositories

Cloned repositories relevant to the project: **Finding Variables Under Consideration using Perplexity-based Probing of LLMs**. All repos support the study of how LLMs encode pragmatic, scalar, and contextual meaning — and how probability-based (perplexity / forced-decoding) signals can reveal variables under consideration.

---

## 1. `scalar-adj-probing/`

**Source:** https://github.com/fangru-lin/llm_scalar_adj  
**Paper:** Lin et al. (2024) — *Probing Large Language Models for Scalar Adjective Lexical Semantics and Scalar Diversity Pragmatics*  
**Venue:** Not yet stated in README; ACL-adjacent

### What it does
Probes LLMs for two related phenomena:
- **Lexical semantics of scalar adjectives** (intensity ranking, scale membership): e.g., whether a model knows that *scalding* > *hot* > *warm*
- **Scalar diversity pragmatics**: predicts which adjectives in a scale are more likely to trigger scalar implicatures (QUD-sensitive phenomena)

### Key scripts
| Script | Purpose |
|--------|---------|
| `lexical_semantics/intensity/direct_intensity_ranking.py` | Embedding-based intensity ranking using cosine similarity |
| `lexical_semantics/intensity/indirect_intensity_ranking.py` | Ranking via next-token prediction / masked LM |
| `lexical_semantics/membership/direct_scale_classification.py` | Embedding-based scale membership classification |
| `lexical_semantics/membership/indirect_scale_classification.py` | Forced-choice log-prob / masked prediction for scale membership |
| `scalar_diversity/predict_diversity.py` | Predicts scalar diversity ratings; uses `get_yes_no_prob()` forced-decoding with token probability extraction |
| `scalar_diversity/predict_diversity_baseline.py` | Baseline comparison |

### Relevance to our project
**Direct.** The `get_yes_no_prob()` pattern — extracting token-level probabilities for Yes/No under different prompts — is exactly the forced-decoding / perplexity-probing paradigm we are studying. The scalar diversity task is essentially measuring which variables (scale alternatives) are under consideration for a given adjective.

### Dependencies
`transformers`, `torch`, `numpy`, `scipy`, `openai`, `scikit-learn`, `tenacity`, `beautifulsoup4`

---

## 2. `scalar-implicature/`

**Source:** https://github.com/joyennn/scalar-implicature  
**Paper:** Cho & Kim (2024) — *Pragmatic inference of scalar implicature by LLMs*  
**Venue:** ACL Student Research Workshop 2024 — https://aclanthology.org/2024.acl-srw.2/

### What it does
Investigates whether BERT and GPT-2 can perform pragmatic inference for scalar implicatures (e.g., interpreting *some* as implying *not all*). Two experiments:
1. **Cosine similarity** — probes internal representations for pragmatic vs. semantic readings
2. **Next-sentence / next-token prediction** — tests whether context (Question Under Discussion) shifts the model toward the pragmatic reading

Key finding: BERT defaults to the pragmatic reading (Default model, Levinson 2000); GPT-2 struggles when context requires active inference (Context-driven model, Sperber & Wilson 2002).

### Key scripts
| File | Purpose |
|------|---------|
| `scalar_implicature.ipynb` | Main Jupyter notebook; all experiments |

### Relevance to our project
Tests exactly the phenomenon of interest: what scalar alternatives are under consideration for *some* vs. *all*. The next-token prediction method directly operationalizes perplexity-based forced decoding. The QUD manipulation is a close analog to our variable-manipulation paradigm.

### Dependencies
`transformers` (BERT, GPT-2), `torch`, standard Python stack

---

## 3. `context-probing/`

**Source:** https://github.com/cifkao/context-probing  
**Paper:** Cífka & Liutkus (2023) — *Black-box language model explanation by context length probing*  
**Venue:** ACL 2023 (Short Papers) — https://aclanthology.org/2023.acl-short.92/

### What it does
Measures how each context token contributes to predicting each target token by systematically varying context length and computing **cross-entropy** and **KL divergence** between predictions at different context lengths. Produces a token × token importance matrix showing which context tokens are most predictive of each output.

### Key scripts / API
| Component | Purpose |
|-----------|---------|
| `context_probing/core.py` — `run_probing()` | Main API: given a model + tokenizer + inputs, returns `xent` and `kl_div` tensors across all context lengths |
| `context_probing/core.py` — `get_delta_scores()` | Converts raw metrics to differential importance scores |
| `context_probing/unigram.py` — `estimate_unigram_logprobs()` | Estimates null-context (unigram) baseline |
| `context_probing/scripts/predict_sliding` | Applies causal LM with sliding window; saves logits as NumPy arrays |
| `context_probing/scripts/preds_to_metrics` | Converts saved logits to xent/KL metrics |
| `process_metrics.ipynb` | Reproduces paper figures |

### Relevance to our project
**Very high.** This repo provides the core machinery for perplexity-based probing. The KL divergence between P(next token | full context) and P(next token | truncated context) is a direct measure of how much information each context variable contributes — i.e., what is "under consideration." The `run_probing()` API can be applied directly to probe which discourse referents / scalar alternatives matter for upcoming token predictions.

### Dependencies
`pip install context-probing` or `poetry install` from lock file. Requires `transformers`, `torch`, `datasets`, `conllu`.

---

## 4. `pragmatic-theories-enhance/`

**Source:** https://github.com/takuma1229/pragmatic-theories-enhance  
**Paper:** Sato, Kawano & Yoshino (2025) — *Pragmatic Theories Enhance Understanding of Implied Meanings in LLMs*  
**Venue:** IJCNLP-AACL 2025 — https://arxiv.org/abs/2510.26253

### What it does
Tests whether explicitly prompting LLMs with descriptions of pragmatic theories (Gricean maxims, RSA, QUD) improves their ability to understand implied meanings. Uses the **PRAGMEGA benchmark** (sourced from https://github.com/jennhu/lm-pragmatics) covering scalar implicature, indirect speech acts, and other pragmatic phenomena. Supports local HuggingFace models and OpenAI API.

### Key scripts
| Script | Purpose |
|--------|---------|
| `scripts/run_inference.py` | Run inference on PRAGMEGA with chosen model and backend |
| `scripts/reproduce.py` | Full reproduction pipeline |
| `scripts/evaluate.py` | Compute metrics from predictions |
| `scripts/make_table.py` | Aggregate results into comparison table |
| `src/modeling.py` | Backend implementations (transformers, openai, dummy) |
| `src/prompts.py` | Prompt templates including theory-augmented prompts |
| `src/metrics.py` | Evaluation metrics |

### Setup
```bash
make setup       # creates uv venv and installs deps
make download_data
make reproduce_main MODEL=<model> BACKEND=transformers
```

### Relevance to our project
Directly tests pragmatic inference in LLMs with theory-guided prompting. The prompt engineering around RSA and QUD is directly relevant to our variable-under-consideration framing. Code can be adapted to insert explicit "variable salience" descriptions into prompts and measure effect on forced-choice token probabilities.

### Dependencies
`torch`, `transformers`, `pandas`, `tqdm`; optional `openai` for API backend. Uses `uv` for environment management.

---

## 5. `llm-salience/`

**Source:** https://github.com/jantrienes/llm-salience  
**Paper:** Trienes et al. (2025) — *Behavioral Analysis of Information Salience in Large Language Models*  
**Venue:** Findings of ACL 2025 — https://aclanthology.org/2025.findings-acl.1204/

### What it does
Introduces a **behavioral probing framework** for information salience: uses length-controlled summarization as a probe of what LLMs consider important. Traces **answerability of Questions Under Discussion (QUDs)** through summaries to derive a salience map. Tests 13 models on 4 datasets (PubMed RCTs, arXiv astrophysics, NLP papers, QMSum meetings).

Key finding: LLMs have a consistent, hierarchical notion of salience — but it cannot be accessed via introspection (direct prompting).

### Key scripts
| Script | Purpose |
|--------|---------|
| `src/info_salience/summarization.py` | Length-controlled summary generation (API and vLLM) |
| `src/info_salience/qa.py` | QUD generation and question answering |
| `src/info_salience/claim_entailment.py` | Claim entailment for answerability scoring |
| `src/info_salience/introspection.py` | Direct introspection / perceived salience probing |
| `notebooks/20-qgen.ipynb` | Question generation and clustering |
| `notebooks/30-salience.ipynb` | Salience map construction |

### Relevance to our project
The QUD-answerability tracing is a behavioral analog to measuring "variables under consideration": as context length or summary length changes, which QUDs remain answerable? This is essentially a perplexity-free version of our approach. The contrast between behavioral (observed) and introspective (perceived) salience is a crucial distinction for our probing paradigm.

### Dependencies
`conda` environment with Python 3.11; `pip install -r requirements-lock.txt`. Uses `vllm` for local model inference, `litellm` for API models.

---

## 6. `rsa-pragmatics/`

**Source:** https://github.com/mawilson1234/RSA  
**Paper:** Based on Cohn-Gordon, Goodman & Potts (2019) — utterance-level and incremental RSA

### What it does
Pure Python implementation of the Rational Speech Act (RSA) model. Supports:
- **Utterance-level** predictions (standard RSA)
- **Incremental** predictions (word-by-word pragmatic inference)
- **Partial utterance** predictions (single-step lookahead)
- Configurable rationality parameter `alpha`

Input is a CSV specifying worlds/objects, words, utterances, and prior probabilities. Output is a posterior distribution over world states given an utterance.

### Key files
| File | Purpose |
|------|---------|
| `rsa.py` | Main RSA implementation — utterance-level and incremental |
| `rsa-comb-data.csv` | Example CSV following Cohn-Gordon et al. (2019) |
| `rsa-comb-data2.csv` | Second example dataset |
| `predictions.txt` | Pre-computed predictions |

### Relevance to our project
The RSA framework is the theoretical backbone for variables-under-consideration: the literal listener and pragmatic listener maintain distributions over world states (= candidate variable values). The incremental mode is especially relevant — it models how prior context shifts which world states remain under consideration at each token. Can be used to generate RSA-predicted probability distributions as a baseline or comparison for LLM-derived perplexity measurements.

### Dependencies
Pure Python (no external packages required for the RSA model itself).

---

## 7. `rsa-latent/`

**Source:** https://github.com/guyemerson/rsa-latent  
**Paper/Notes:** Emerson (ESSLLI 2016) — Alternative frameworks for gradable predicates in RSA

### What it does
Implements two alternative RSA models for **gradable predicates** (scalar adjectives like *expensive*, *tall*) in [WebPPL](http://webppl.org/):
1. **Permanently lifted variables**: latent threshold unobserved at all levels (avoids intractability)
2. **Inherently vague predicates**: truth as a random variable (conditionally independent, tractable)

Contrasts with the standard RSA approach where the pragmatic listener "lifts" a latent threshold variable.

### Key files
| File | Purpose |
|------|---------|
| `rsa_gradable_latent.wppl` | Permanently lifted variable model (WebPPL) |
| `rsa_gradable_vague.wppl` | Inherently vague predicate model (WebPPL) |
| `rsa_gradable_original.wppl` | Original RSA gradable predicate model for comparison |

### Relevance to our project
Directly addresses the theoretical question of how scalar adjective thresholds are represented as latent variables in probabilistic models. The "permanently lifted" approach is analogous to treating scalar variables as always under consideration (marginalized over by the listener), while the "vague predicate" approach treats degree as a property of word meaning itself. Both framings are relevant for interpreting what LLM perplexity signals about scalar variable representation.

### Dependencies
Requires [WebPPL](http://webppl.org/) (probabilistic programming language running in Node.js).

---

## 8. `crsa-llm/`

**Source:** https://github.com/LautaroEst/crsa  
**Paper:** Estienne et al. (2025) — *Collaborative Rational Speech Act: Pragmatic Reasoning for Multi-Turn Dialog*  
**Venue:** EMNLP 2025 — https://aclanthology.org/2025.emnlp-main.1145/

### What it does
Introduces **Collaborative RSA (CRSA)**: an information-theoretic extension of RSA for multi-turn, collaborative dialogue. Unlike standard RSA (single-turn reference games), CRSA:
- Models both agents as having private information
- Optimizes a gain function from rate-distortion theory
- Supports LLM integration (tested with Llama-3.2-1B-Instruct on MDDial medical dataset)

### Key scripts
| Script | Purpose |
|--------|---------|
| `src/crsa/scripts/naive_reference_game.py` | Reference game demonstration with plots |
| `src/crsa/scripts/run_mddial.py` | Run Llama on medical dialog dataset; produce literal speaker values |

### Relevance to our project
CRSA extends RSA to the scenario where variables under consideration span multiple turns — relevant when the "variable" is discourse-level context rather than a single lexical choice. The literal speaker probability values extracted from LLMs (used as input to CRSA) are directly analogous to our forced-decoding perplexity signals. This repo shows how to bridge LLM token probabilities with RSA-style pragmatic inference.

### Dependencies
`conda` with Python 3.12.7; `pip install -r requirements.txt && pip install -e .`. Requires GPU for LLM inference (NVIDIA V100 32GB used in paper).

---

## Summary Table

| Repo | Method | Phenomenon | LLM Integration | Perplexity/Forced-Decoding |
|------|--------|-----------|-----------------|---------------------------|
| `scalar-adj-probing` | Embedding + forced-decoding | Scalar adjective intensity & diversity | BERT, GPT-2, GPT-4, T5 | Yes — `get_yes_no_prob()` |
| `scalar-implicature` | Cosine similarity + next-token | Scalar implicature (*some* → *not all*) | BERT, GPT-2 | Yes — next-token prediction |
| `context-probing` | KL divergence over context lengths | Context token importance | Any causal LM (HuggingFace) | **Core method** — perplexity probing |
| `pragmatic-theories-enhance` | Prompted inference | General pragmatic understanding | Any HF model or OpenAI | No — generative scoring |
| `llm-salience` | Length-controlled summarization + QUD | Information salience | 13 models (vLLM/API) | Indirect — behavioral probe |
| `rsa-pragmatics` | RSA computation | Reference games, pragmatic inference | None (pure RSA) | No — probabilistic model |
| `rsa-latent` | RSA with latent variables (WebPPL) | Scalar/gradable adjective thresholds | None (WebPPL) | No — theoretical model |
| `crsa-llm` | CRSA (rate-distortion RSA) | Multi-turn collaborative dialog | Llama-3.2 | Partial — literal speaker probs |

---

## Recommended Integration Path

For probing **variables under consideration** using perplexity:

1. **Start with `context-probing`** (`run_probing()` API) — use KL divergence to measure which context tokens (candidate variable values) most influence upcoming token predictions.

2. **Use `scalar-adj-probing`** as the task domain — its `get_yes_no_prob()` pattern is the forced-decoding probe; its datasets (scalar scales, diversity ratings) define the variable space.

3. **Validate against `scalar-implicature`** — the QUD manipulation there tests exactly what shifts the probability distribution over alternatives.

4. **Use `rsa-pragmatics`** (and `rsa-latent` for gradable adjectives) to generate theoretical RSA predictions as a computational baseline to compare against LLM perplexity signals.

5. **Use `llm-salience`** methodology (behavioral vs. introspective probing) to test whether forced-decoding signals match model self-reports.

6. **`pragmatic-theories-enhance`** and **`crsa-llm`** show how theory-guided prompting and multi-turn RSA can modulate the probability signals — useful for ablation studies.
