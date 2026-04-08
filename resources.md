# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Finding Variables Under Consideration" — investigating whether perplexity-based probing can reveal which variables LLMs actually consider during message generation.

## Papers
Total papers downloaded: 26

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Probing LLMs for Scalar Adjective Lexical Semantics and Scalar Diversity Pragmatics | Lin, Altshuler, Pierrehumbert | 2024 | `papers/2404.03301_probing_LLMs_scalar_adjective_pragmatics_2024.pdf` | Perplexity-based probing of scalar adjectives; core methodology |
| 2 | Pragmatic Inference of Scalar Implicature by LLMs | Cho, Kim | 2024 | `papers/2408.06673_pragmatic_inference_scalar_implicature_LLMs_2024.pdf` | Surprisal under QUD manipulation |
| 3 | Black-box LM Explanation by Context Length Probing | Cífka, Liutkus | 2023 | `papers/2212.14815_context_length_probing_blackbox_LM_2022.pdf` | KL divergence-based probing framework |
| 4 | Implicature in Interaction | — | 2025 | `papers/2510.25426_implicature_interaction_human_LLM_2025.pdf` | Implicature in human-LLM interaction |
| 5 | Do LLMs Understand Conversational Implicature | — | 2024 | `papers/2404.19509_LLMs_understand_conversational_implicature_2024.pdf` | Chinese sitcom implicature benchmark |
| 6 | The Goldilocks of Pragmatic Understanding | Ruis et al. | 2022 | `papers/2210.14986_goldilocks_pragmatic_understanding_LLMs_2022.pdf` | Comprehensive implicature benchmark (LUDWIG) |
| 7 | PUB: Pragmatics Understanding Benchmark | Hu et al. | 2024 | `papers/2401.07078_PUB_pragmatics_understanding_benchmark_2024.pdf` | Multi-task pragmatics benchmark |
| 8 | Experimental Pragmatics with Machines | — | 2024 | `papers/2405.05776_experimental_pragmatics_machines_2024.pdf` | Disjunction inference testing |
| 9 | The Pragmatic Mind of Machines | — | 2025 | `papers/2505.18497_pragmatic_mind_of_machines_2025.pdf` | Emergence of pragmatic competence |
| 10 | Language Models Identify Ambiguities and Exploit Loopholes | — | 2025 | `papers/2508.19546_LMs_identify_ambiguities_exploit_loopholes_2025.pdf` | Pragmatics through loophole analysis |
| 11 | PragWorld | — | 2025 | `papers/2511.13021_PragWorld_benchmark_2025.pdf` | World model under linguistic alterations |
| 12 | LLM Limits on Implicit Content in Political Discourse | — | 2025 | `papers/2506.06775_LLM_limits_implicit_content_political_discourse_2025.pdf` | Implicature in political language |
| 13 | Information-Theoretic Probing for Linguistic Structure | Pimentel et al. | 2020 | `papers/2004.03061_information_theoretic_probing_2020.pdf` | Foundational probing methodology |
| 14 | Amnesic Probing: Behavioral Explanation with Counterfactuals | Elazar et al. | 2020 | `papers/2006.00995_amnesic_probing_behavioral_explanation_2020.pdf` | Counterfactual probing method |
| 15 | LLM Probabilities Cannot Distinguish Possible/Impossible Language | — | 2025 | `papers/2509.15114_LLM_probabilities_possible_impossible_language_2025.pdf` | Limitations of LLM probability measures |
| 16 | Language Model Behavior: A Comprehensive Survey | — | 2023 | `papers/2303.11504_language_model_behavior_survey_2023.pdf` | Survey of LM behavioral studies |
| 17 | BOSE: Systematic Evaluation for Base Models | — | 2025 | `papers/2503.00812_BOSE_evaluation_base_models_2025.pdf` | Evaluation methods for pre-trained models |
| 18 | QUDSELECT: Selective Decoding for QUD Parsing | — | 2024 | `papers/2408.01046_QUDselect_questions_under_discussion_parsing_2024.pdf` | QUD parsing method |
| 19 | QUDEVAL: Evaluation of QUD Discourse Parsing | — | 2023 | `papers/2310.14520_QUDEVAL_questions_under_discussion_evaluation_2023.pdf` | QUD evaluation framework |
| 20 | Discourse Analysis via QUD | — | 2022 | `papers/2210.05905_discourse_QUD_parsing_dependency_2022.pdf` | QUD dependency parsing |
| 21 | Survey of QUD Models for Discourse Processing | — | 2025 | `papers/2502.15573_survey_QUD_models_discourse_processing_2025.pdf` | Comprehensive QUD survey |
| 22 | Conversational Implicatures: Modelling Relevance Theory Probabilistically | Unger, Buschmeier | 2025 | `papers/2509.22354_conversational_implicatures_relevance_theory_probabilistic_2025.pdf` | Bayesian Relevance Theory model |
| 23 | Pragmatic Theories Enhance Understanding of Implied Meanings in LLMs | Sato et al. | 2025 | `papers/2510.26253_pragmatic_theories_implied_meanings_LLMs_2025.pdf` | Pragmatic theory prompting |
| 24 | Gricean Norms as Basis for Effective Collaboration | — | 2025 | `papers/2503.14484_gricean_norms_effective_collaboration_2025.pdf` | Gricean norms for AI collaboration |
| 25 | Behavioral Analysis of Information Salience in LLMs | Trienes et al. | 2025 | `papers/2502.14613_behavioral_analysis_information_salience_LLMs_2025.pdf` | QUD-based salience probing |
| 26 | QUDsim: Discourse Similarities in LLM Text | — | 2025 | `papers/2504.09373_QUDsim_discourse_similarities_LLM_text_2025.pdf` | QUD for measuring text uniqueness |

See `papers/README.md` for detailed descriptions.

## Datasets
Total dataset collections downloaded: 8

| Name | Source | Size | Task | Location |
|------|--------|------|------|----------|
| BIG-bench Implicatures | Google BIG-bench | 492 examples | Implicature yes/no | `datasets/bigbench_implicatures/` |
| LUDWIG / do-pigs-fly | Ruis et al. 2024 | 629 test examples | Conversational implicature | `datasets/UCL-DARK_ludwig/` |
| tasksource/implicatures | George & Mamidi 2020 | 1,000 examples | Implicature pairs | `datasets/tasksource_implicatures/` |
| LM-Pragmatics | Hu et al. 2022 | 4 sub-tasks | Pragmatic reasoning | `datasets/lm-pragmatics/` |
| Scalar Diversity | Van Tiel et al. 2016 | 43 pairs | SI endorsement rates | `datasets/scalar_diversity/` |
| Scalar Adjectives | Garí Soler & Apidianaki 2020 | 187 pairs | Scale membership/intensity | `datasets/scalar_adjectives/` |
| Stanford Politeness | Danescu-Niculescu-Mizil et al. | ~2,500 examples | Politeness classification | `datasets/stanford_politeness/` |
| PUB Pragmatics | Doshi et al. 2024 | 5,000+ | Multi-task pragmatics | `datasets/PUB_pragmatics/` |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 8

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| Scalar Adj Probing | github.com/fangru-lin/llm_scalar_adj | Perplexity-based scalar adjective probing | `code/scalar-adj-probing/` |
| Scalar Implicature | github.com/joyennn/scalar-implicature | Surprisal-based implicature testing | `code/scalar-implicature/` |
| Context Probing | github.com/cifkao/context-probing | Context length probing with KL divergence | `code/context-probing/` |
| Pragmatic Theories | github.com/takuma1229/pragmatic-theories-enhance | Pragmatic theory prompting | `code/pragmatic-theories-enhance/` |
| LLM Salience | github.com/jantrienes/llm-salience | Information salience via behavioral probing | `code/llm-salience/` |
| RSA Pragmatics | github.com/mawilson1234/RSA | Pure Python RSA implementation | `code/rsa-pragmatics/` |
| RSA Latent | github.com/guyemerson/rsa-latent | RSA with vague predicates (WebPPL) | `code/rsa-latent/` |
| CRSA-LLM | github.com/LautaroEst/crsa | Collaborative RSA for multi-turn dialog | `code/crsa-llm/` |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- **Paper-finder service** with diligent mode for comprehensive semantic search.
- **arXiv API** with 5 targeted queries covering pragmatics, perplexity probing, QUD, implicature, and relevance theory.
- **Semantic Scholar API** for citation-ranked results.
- Manual identification of key papers from references in core papers.

### Selection Criteria
- Papers directly using perplexity/surprisal for probing LLM behavior (highest priority).
- Papers on LLM pragmatic competence, especially scalar implicature and conversational implicature.
- Papers on QUD framework (provides theoretical basis for "variables under consideration").
- Papers on behavioral vs. introspective assessment of LLM capabilities.
- Methodological papers on probing techniques.

### Challenges Encountered
- Van Tiel et al. 2014 raw experimental data not publicly available; reconstructed summary statistics.
- GYAFC formality corpus requires Yahoo ToS agreement — not downloaded.
- Some paper-finder searches timed out; supplemented with direct arXiv API queries.

### Gaps and Workarounds
- No existing dataset specifically designed for "variable-under-consideration" probing — will need to construct experimental scenarios.
- Scalar diversity datasets provide a good starting point but are limited to adjective scales.
- Politeness corpus can serve as an axis-of-variation dataset for testing perplexity sensitivity.

## Recommendations for Experiment Design

### Primary Dataset(s)
1. **LUDWIG/do-pigs-fly** — Established implicature benchmark for evaluating pragmatic understanding.
2. **Scalar adjective datasets** — For testing perplexity-based probing of pragmatic variables.
3. **Custom prompt construction** — Create scenarios with systematically varied axes (formality, politeness, specificity, sentiment) to measure perplexity sensitivity.

### Baseline Methods
1. **Unmodified perplexity** — Base measure without axis manipulation.
2. **Random variation control** — Perplexity changes for semantically irrelevant modifications.
3. **RSA predictions** — Formal pragmatic predictions for what should be under consideration.

### Recommended Metrics
1. **Perplexity ratio** across axis variations.
2. **KL divergence** between distributions under different pre-fills.
3. **Correlation** with human pragmatic judgments.

### Code to Adapt/Reuse
1. **context-probing** — Core KL divergence computation framework.
2. **scalar-adj-probing** — Forced-decoding probability extraction (`get_yes_no_prob()`).
3. **crsa-llm** — Bridge between LLM token probabilities and RSA framework.
4. **llm-salience** — Behavioral probing methodology (length-constrained output analysis).
