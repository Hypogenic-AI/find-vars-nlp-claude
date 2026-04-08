# Literature Review: Finding Variables Under Consideration

## Research Area Overview

This literature review covers research relevant to probing which variables LLMs actually consider when generating messages. The hypothesis is that LLMs do not always consider the pragmatic implications of their suggestions, even though they can identify such implications when prompted. By pre-filling different responses that vary certain axes and measuring perplexity changes, it may be possible to determine which variables the LLM is "attending to" during generation.

The relevant literature spans three interconnected areas: (1) LLM pragmatic competence and implicature, (2) probing and explanation methods using perplexity/surprisal, and (3) Questions Under Discussion (QUD) and information salience frameworks.

---

## Key Papers

### 1. Lin, Altshuler & Pierrehumbert (2024) — "Probing LLMs for Scalar Adjective Lexical Semantics and Scalar Diversity Pragmatics"
- **Source**: arXiv 2404.03301
- **Key Contribution**: Probes LLMs (BERT, RoBERTa, GPT-4, Flan-T5, Falcon) for scalar adjective knowledge using both direct (embedding-based) and indirect (perplexity-based) methods.
- **Methodology**: **Perplexity comparison of minimal pairs** — compares perplexity of correct vs. incorrect scalar constructions (e.g., "good but not awesome" vs. "awesome but not good"). Lower perplexity for correct order = model encodes intensity knowledge.
- **Datasets**: DEMELO, CROWD, WILKINSON scalar adjective datasets; ukWac context sentences.
- **Key Results**: LLMs encode rich lexical-semantic info about scalar adjectives but have **unsatisfying performance on scalar diversity** (pragmatic aspect). Larger models are not always better. GPT-4 best overall.
- **Relevance**: **Directly demonstrates the perplexity-based probing methodology** central to our hypothesis. Shows that semantic knowledge ≠ pragmatic competence.
- **Code**: https://github.com/fangru-lin/llm_scalar_adj

### 2. Cho & Kim (2024) — "Pragmatic Inference of Scalar Implicature by LLMs"
- **Source**: arXiv 2408.06673
- **Key Contribution**: Tests whether BERT and GPT-2 engage in pragmatic inference of "some" → "not all" implicature, and whether QUD context modulates this.
- **Methodology**: (1) Cosine similarity between "some"-sentences and semantic/pragmatic paraphrases; (2) **Surprisal measurement** under different QUD contexts (upper-bound "all" vs. lower-bound "any" questions).
- **Datasets**: 198 sentences from BNC with "some + NP" structures.
- **Key Results**: Both models prefer pragmatic interpretation without context. GPT-2 shows **higher surprisal when pragmatic inference is required** (upper-bound QUD), suggesting processing difficulty. BERT is insensitive to QUD type.
- **Relevance**: **Directly uses surprisal/perplexity as a probe for pragmatic processing**, and manipulates context (QUD) to test variable sensitivity — very close to our proposed approach.
- **Code**: https://github.com/joyennn/scalar-implicature

### 3. Cífka & Liutkus (2023) — "Black-box Language Model Explanation by Context Length Probing"
- **Source**: arXiv 2212.14815
- **Key Contribution**: Novel explanation technique for causal LMs based on tracking predictions as a function of context length, assigning **differential importance scores** (Δ-scores).
- **Methodology**: Runs model on sliding windows of varying context length c. Computes KL divergence D(c) = DKL[P_max || P_c] to quantify information loss from shorter context. **Δ-score** = change in KL when adding one more token to context.
- **Datasets**: English LinES treebank (Universal Dependencies).
- **Key Results**: Power-law-like inverse relationship between context length and importance. Proper nouns are most informative in distant context. Larger models rely more on shorter contexts.
- **Relevance**: **Core methodological inspiration** — demonstrates how to use probability distributions over the vocabulary (not just single-token likelihood) to probe what the model "considers." Model-agnostic, black-box approach.
- **Code**: https://github.com/cifkao/context-probing

### 4. Sato, Kawano & Yoshino (2025) — "Pragmatic Theories Enhance Understanding of Implied Meanings in LLMs"
- **Source**: arXiv 2510.26253
- **Key Contribution**: Providing pragmatic theory summaries (Grice, Relevance Theory) as prompts improves LLM performance on implied meaning tasks by up to 9.6%.
- **Methodology**: Zero-shot prompting with theory descriptions; tests on multiple pragmatic reasoning benchmarks.
- **Key Results**: Even just mentioning theory names (without full descriptions) improves performance by 1-3% in larger models. Particularly effective for maxim-flouting utterances and irony.
- **Relevance**: Shows LLMs have latent pragmatic knowledge that can be activated — supports the idea that models "know" about pragmatic variables but don't always "use" them during generation.

### 5. Trienes, Schlötterer, Li & Seifert (2025) — "Behavioral Analysis of Information Salience in LLMs"
- **Source**: arXiv 2502.14613
- **Key Contribution**: Uses **length-constrained summarization as a behavioral probe** into LLMs' content selection, tracked via QUD answerability.
- **Methodology**: Generate summaries at different length budgets (10-200 words). Track which QUDs remain answerable at each length → Content Salience Maps. 13 models, 4 datasets.
- **Key Results**: LLMs have a **nuanced, hierarchical notion of salience** consistent across model families. **Salience cannot be accessed through introspection** (asking models directly gives different answers than behavior shows). Weak correlation with human salience perception.
- **Relevance**: **Critical methodological parallel** — uses behavioral probing (output variation) rather than introspection to determine what models "consider important." The finding that introspection ≠ behavior directly motivates our perplexity-based probing approach.
- **Code**: https://github.com/jantrienes/llm-salience

### 6. Unger & Buschmeier (2025) — "Conversational Implicatures: Modelling Relevance Theory Probabilistically"
- **Source**: arXiv 2509.22354
- **Key Contribution**: Bayesian probabilistic model of implicature comprehension implementing Relevance Theory in ProbLog.
- **Methodology**: Formalizes Relevance Theory concepts (manifestness, cognitive effects, processing effort) probabilistically. Models how listeners derive implicatures through intention attribution.
- **Relevance**: Provides formal probabilistic framework connecting pragmatic theory to computational models — could inform how to formalize "variables under consideration" in terms of cognitive effects and manifestness.

### 7. Ruis et al. (2024) — "The Goldilocks of Pragmatic Understanding"
- **Source**: arXiv 2210.14986
- **Key Contribution**: Comprehensive benchmark testing LLMs on conversational implicature understanding. GPT-4 achieves 88.7% with 30-shot CoT prompting.
- **Datasets**: "do-pigs-fly" / LUDWIG dataset — 629 test examples of conversational implicatures.
- **Relevance**: Provides primary benchmark dataset for evaluating pragmatic understanding, and establishes that instruction tuning and CoT are needed for good pragmatic performance.

### 8. Hu et al. (2022, 2023) — Scalar Diversity and Surprisal
- **Key Contribution**: Uses **string-based and concept-based surprisal** from GPT-2 to predict scalar diversity ratings.
- **Methodology**: Measures surprise of a strong word appearing as alternative to a weak word. Concept-based surprisal averages over semantically similar alternatives.
- **Key Results**: Scalar diversity correlates with both surprisal measures. Stronger correlations than many feature-based predictors.
- **Relevance**: Demonstrates that **surprisal/perplexity can capture pragmatic phenomena** (scalar diversity) that simpler features miss.

### 9. QUD Framework Papers (2022-2025)
- **Discourse Analysis via QUD** (2210.05905): QUD parsing as dependency structure.
- **QUDEVAL** (2310.14520): Evaluation framework for QUD discourse parsing.
- **QUDSELECT** (2408.01046): Selective decoding for QUD parsing.
- **Survey of QUD Models** (2502.15573): Comprehensive survey of QUD applications.
- **Relevance**: QUD provides the theoretical framework for understanding what is "under consideration" in a discourse. A variable is "under consideration" if it relates to the current QUD.

### 10. Additional Pragmatics + LLM Papers
- **PUB** (2401.07078): Pragmatics Understanding Benchmark covering implicature, presupposition, deixis.
- **Gricean Norms as Basis for Collaboration** (2503.14484): Gricean maxims for human-AI collaboration.
- **PragWorld** (2511.13021): Benchmark evaluating LLMs' world models under minimal linguistic alterations.
- **Pragmatic Mind of Machines** (2505.18497): Traces emergence of pragmatic competence in LLMs.
- **LLM Limits on Implicit Content** (2506.06775): LLMs struggle with implicit content in political discourse.

---

## Common Methodologies

### Perplexity/Surprisal-Based Probing
- **Minimal pair comparison**: Compare perplexity of correct vs. incorrect constructions (Lin et al., 2024; Hu et al., 2022).
- **Surprisal under context manipulation**: Vary context (e.g., QUD type) and measure surprisal changes in a fixed response (Cho & Kim, 2024).
- **Context length probing**: Track how predictions change as context length varies (Cífka & Liutkus, 2023).
- **Self-aligned perplexity**: Measure how well outputs match a model's own "style" (Ren et al., 2025).

### Direct vs. Indirect Probing
- **Direct probing**: Analyze hidden representations/embeddings (requires open-source models).
- **Indirect probing**: Use behavioral tests — prompting, generation, perplexity comparison (works for all models).
- Key finding: indirect probing underperforms direct probing but reveals valid relative rankings (Lin et al., 2024).

### Behavioral vs. Introspective Assessment
- **Behavioral**: What the model actually does (generation, perplexity, etc.).
- **Introspective**: What the model says it would do when asked directly.
- Critical finding: **these diverge** — models' self-reported salience doesn't match their actual behavior (Trienes et al., 2025).

---

## Standard Baselines
- **Random/majority baseline**: For classification tasks (implicature yes/no).
- **Google Ngram**: Frequency-based baseline for perplexity comparisons (Lin et al., 2024).
- **Fast-text embeddings**: Static embedding baseline for probing tasks.
- **Zero-shot prompting**: Standard LLM baseline without pragmatic guidance.
- **Chain-of-Thought (CoT)**: Step-by-step reasoning baseline.
- **RSA (Rational Speech Act)**: Formal pragmatic baseline from computational linguistics.

---

## Evaluation Metrics
- **Perplexity / Surprisal**: Core metric for probing — measures how "surprised" the model is by a given continuation.
- **KL Divergence**: Measures distributional shift between conditions (Cífka & Liutkus, 2023).
- **Pairwise Accuracy**: Whether correct ordering/interpretation has lower perplexity than incorrect.
- **Cosine Similarity**: Between sentence embeddings for semantic/pragmatic alignment.
- **MRR (Mean Reciprocal Rank)**: For scale membership probing.
- **Correlation**: Between model perplexity patterns and human judgments (scalar diversity ratings).

---

## Datasets in the Literature
| Dataset | Used By | Task | Size |
|---------|---------|------|------|
| DEMELO, CROWD, WILKINSON | Lin et al. 2024; Garí Soler & Apidianaki 2020 | Scalar adjective probing | 185 half-scales |
| ukWac context sentences | Lin et al. 2024 | Context for SA probing | 10 per half-scale |
| BNC "some" sentences | Cho & Kim 2024 | Scalar implicature | 198 sentences |
| do-pigs-fly / LUDWIG | Ruis et al. 2024 | Conversational implicature | 629 test |
| BIG-bench Implicatures | Google | Implicature classification | 492 examples |
| PUB | Hu et al. 2024 | Pragmatics benchmark | 5,000+ |
| PRAGMEGA | Sato et al. 2025 | Pragmatic reasoning | Multi-task |
| LinES treebank | Cífka & Liutkus 2023 | Context probing | 20K tokens |
| Van Tiel et al. 2016 | Multiple | Scalar diversity ratings | 43 pairs |

---

## Gaps and Opportunities

1. **No work directly probes which variables LLMs "consider" during generation** using systematic perplexity measurement across response variations. Existing work probes comprehension, not production.

2. **Pre-filling / forced decoding as probing tool** is underexplored. While forced decoding is used in translation and evaluation, it hasn't been systematically used to probe which axes of variation the model is sensitive to.

3. **The gap between model knowledge and model behavior** (identified by Trienes et al. 2025 and Lin et al. 2024) is precisely what our hypothesis targets — models "know" about pragmatic variables but don't always "use" them.

4. **Cross-axis comparison** — comparing sensitivity across different variable types (formality, politeness, sentiment, specificity, etc.) — has not been done in a unified framework.

5. **Connection between QUD framework and LLM generation** — QUDs provide a principled way to define what "should" be under consideration, but this hasn't been connected to perplexity-based probing of generation behavior.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **LUDWIG / do-pigs-fly** (Ruis et al. 2024) — Primary implicature benchmark with rich context.
2. **BIG-bench Implicatures** — Simple yes/no implicature task for baseline evaluation.
3. **Scalar adjective datasets** (DEMELO/CROWD/WILKINSON) — For scalar diversity experiments.
4. **Van Tiel et al. scalar diversity ratings** — Human judgments for correlation analysis.
5. **Custom prompt sets** — Create scenarios where responses vary along specific axes (formality, politeness, directness, etc.) and measure perplexity differences.

### Recommended Baselines
1. **Perplexity of unmodified responses** — Baseline for comparison.
2. **Random axis variation** — Control condition showing perplexity changes for irrelevant modifications.
3. **RSA predictions** — Formal pragmatic baseline for what "should" be under consideration.

### Recommended Metrics
1. **Perplexity difference** (or ratio) across axis variations — primary measure.
2. **KL divergence** between output distributions under different pre-fills.
3. **Statistical significance** of perplexity differences (paired tests across items).
4. **Correlation with human judgments** of variable relevance.

### Methodological Considerations
- Use **multiple models** (open-source: Llama, Mistral; closed: GPT-4) to test generalizability.
- Use **multiple prompt contexts** to avoid confounds.
- Control for **surface-level changes** (token count, word frequency) vs. pragmatic changes.
- The **context-probing** codebase provides a ready-made framework for computing KL divergence over vocabulary distributions.
- The **scalar-adj-probing** codebase demonstrates forced-decoding probability extraction.
