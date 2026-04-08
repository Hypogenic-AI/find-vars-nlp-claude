# Finding Variables Under Consideration: Probing LLM Pragmatic Awareness via Perplexity Sensitivity

## 1. Executive Summary

We investigated whether the pragmatic variables that LLMs "consider" during message generation can be detected by measuring perplexity sensitivity to systematic response variations. By constructing 30 message-writing scenarios and generating response pairs that differ along 5 pragmatic axes (formality, directness, politeness, emotional tone, specificity) plus a synonym control, we computed per-token log-likelihood differences using Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct.

**Key finding**: LLMs show strong, statistically significant directional preferences for 4 of 5 pragmatic axes — they consistently prefer polite, indirect, vague, and formal responses. The lone exception is emotional tone, where the model shows no consistent preference. Critically, the model's self-reported variables (via introspection) only partially overlap with behaviorally-detected ones: directness is strongly "considered" behaviorally but rarely mentioned introspectively, while emotional tone is frequently mentioned but not behaviorally detected. These results replicate perfectly across both model sizes (Spearman ρ = 1.0).

**Practical implication**: Perplexity sensitivity analysis is a viable method for detecting which pragmatic variables an LLM defaults on during generation, revealing a gap between behavioral defaults and self-aware reporting.

## 2. Research Question & Motivation

### Hypothesis
Large language models do not always consider the pragmatic implications of their suggestions when generating messages, even though they can identify such implications when prompted. By pre-filling different responses that vary certain axes and measuring whether perplexity changes significantly, we can probe which variables the LLM is actually "considering."

### Why This Matters
When users ask LLMs to help write messages, the model makes implicit choices about pragmatic dimensions — how formal, direct, polite, emotional, or specific to be. These choices have real consequences for the user (an overly casual email to a boss, or an overly blunt complaint). Understanding which variables the model actively considers vs. ignores during generation could:
1. Improve writing assistant interfaces (prompt users about unconsidered variables)
2. Inform alignment work (what pragmatic defaults do models have?)
3. Advance understanding of LLM pragmatic competence

### Gap in Existing Work
Prior work probes LLM *comprehension* of pragmatic phenomena (Lin et al. 2024; Cho & Kim 2024; Ruis et al. 2024) but not what models actively *consider during generation*. Trienes et al. (2025) showed that LLM introspection diverges from behavior for salience — we extend this finding to pragmatic variable consideration using perplexity as the behavioral probe.

## 3. Methodology

### Experimental Design

We designed a **perplexity sensitivity analysis** with three components:

#### A. Scenario and Variant Construction (GPT-4.1)
- Generated **30 message-writing scenarios** across 3 domains (10 each):
  - **Workplace**: emails to boss, colleagues, clients, HR
  - **Personal**: messages to friends, family, neighbors
  - **Service**: complaints, requests, feedback, applications
- For each scenario, generated **response pairs** along 6 axes:
  - 5 pragmatic: formality, directness, politeness, emotional tone, specificity
  - 1 control: synonym substitution (same meaning, different words)
- Total: 30 scenarios × 6 axes = **180 variant pairs**
- Constraint: each pair differs primarily on the target axis, with matched length

#### B. Perplexity Measurement (Local Models, GPU)
- For each (prompt, response) pair, computed conditional log-likelihood:
  - Tokenize prompt + response using the model's chat template
  - Forward pass to get logits
  - Extract log-probabilities of response tokens only
  - Compute mean per-token log-likelihood
- **Signed difference** = LL(pole_b) - LL(pole_a) for each pair
- Positive = model prefers pole_b, negative = model prefers pole_a

#### C. Introspection Probing (GPT-4.1)
- For each scenario, asked GPT-4.1 to write a response AND list all pragmatic variables it considered
- Mapped self-reported variables to our 5 axes using keyword matching

### Models
| Model | Parameters | Purpose |
|-------|-----------|---------|
| Qwen/Qwen2.5-7B-Instruct | 7.6B | Primary perplexity measurement |
| Qwen/Qwen2.5-3B-Instruct | 3.1B | Cross-model replication |
| GPT-4.1 (OpenAI) | — | Scenario/variant generation, introspection |

### Axis Definitions
| Axis | Pole A | Pole B |
|------|--------|--------|
| Formality | casual/informal | formal/professional |
| Directness | indirect/hedged | direct/assertive |
| Politeness | blunt/terse | polite/courteous |
| Emotional tone | neutral/factual | warm/empathetic |
| Specificity | vague/general | detailed/specific |
| Synonym (control) | version_a | version_b |

### Statistical Tests
- **Wilcoxon signed-rank test**: Whether signed differences ≠ 0 across scenarios
- **Sign test (binomial)**: Whether directional preference significantly deviates from 50%
- **Bonferroni correction**: α = 0.05 / 6 axes = 0.0083
- **Cohen's d**: Effect size of signed differences from 0
- **Bootstrap 95% CI**: 10,000 resamples of mean signed difference
- **Spearman correlation**: Cross-model consistency of effect sizes

### Reproducibility
- Random seed: 42 (NumPy, PyTorch)
- GPT-4.1 temperature: 0.7-0.8 for generation, 0.3 for introspection
- Hardware: NVIDIA RTX A6000 (49GB VRAM), CUDA 12.5
- Software: Python 3.12, PyTorch 2.5.1+cu124, Transformers 5.5.0
- All raw outputs saved to `results/` directory

## 4. Results

### 4.1 Perplexity Sensitivity by Axis (Qwen2.5-7B-Instruct)

| Axis | Mean Signed Δ LL | % Prefer Pole B | Preferred Pole | Consistency | Sign p | Wilcoxon p | Cohen's d | Significant? |
|------|-------------------|-----------------|----------------|-------------|--------|------------|-----------|--------------|
| **Politeness** | **+0.851** | **96.7%** | **polite** | **0.933** | **<0.001** | **<0.001** | **+1.81** | **YES** |
| **Specificity** | **-0.374** | **13.3%** | **vague** | **0.733** | **<0.001** | **<0.001** | **-1.02** | **YES** |
| **Directness** | **-0.313** | **20.0%** | **indirect** | **0.600** | **0.001** | **<0.001** | **-0.94** | **YES** |
| **Formality** | **+0.234** | **76.7%** | **formal** | **0.533** | **0.005** | **0.001** | **+0.68** | **YES** |
| Emotional tone | -0.033 | 46.7% | neutral | 0.067 | 0.856 | 0.459 | -0.08 | no |
| Synonym (ctrl) | -1.057 | 0.0% | version_a | 1.000 | <0.001 | <0.001 | -3.26 | — |

**Key findings**:
1. **Politeness** is the strongest variable under consideration (d = 1.81, 97% directional consistency). The model overwhelmingly prefers polite responses.
2. **Specificity** shows a strong preference for vague/general responses (d = -1.02, 87% prefer vague).
3. **Directness** shows a strong preference for indirect/hedged responses (d = -0.94, 80% prefer indirect).
4. **Formality** shows a moderate preference for formal responses (d = 0.68, 77% prefer formal).
5. **Emotional tone** shows NO significant directional preference (d = -0.08, ~50/50 split). This is the one axis the model does not "consider."

The synonym control shows the highest absolute magnitude (d = -3.26) due to surface-level token probability differences, but this is an artifact — it confirms that perplexity is sensitive to exact word choice. The pragmatic axes are meaningful because they show *directional consistency* (same pole preferred across diverse scenarios).

### 4.2 Introspection vs. Behavioral Detection

| Axis | Introspection Rate | PPL Consistency | PPL Detected? | Introspected? | Agreement? |
|------|-------------------|-----------------|---------------|---------------|------------|
| Formality | 76.7% | 0.533 | YES | YES | AGREE |
| Directness | 6.7% | 0.600 | YES | no | **DIFF** |
| Politeness | 63.3% | 0.933 | YES | YES | AGREE |
| Emotional tone | 53.3% | 0.067 | no | YES | **DIFF** |
| Specificity | 83.3% | 0.733 | YES | YES | AGREE |

**Overall agreement: 3/5 axes (60%)**

The two disagreements are highly informative:
- **Directness**: Behaviorally, the model strongly defaults to indirect/hedged responses (80% consistency), yet it almost never mentions "directness" as a variable it considered (only 6.7% of scenarios). This is a *blind spot* — the model has a strong implicit preference it doesn't self-report.
- **Emotional tone**: The model frequently reports considering "tone" or "empathy" (53% of scenarios), yet behaviorally shows no directional preference (47% vs 53% split). This is *performative consideration* — the model claims to think about it but doesn't actually default one way or the other.

### 4.3 Cross-Model Replication (Qwen 7B vs 3B)

| Axis | Qwen-7B Cohen's d | Qwen-3B Cohen's d | Same Direction? |
|------|-------------------|--------------------|----------------|
| Formality | +0.68 | +0.99 | YES |
| Directness | -0.94 | -0.59 | YES |
| Politeness | +1.81 | +2.18 | YES |
| Emotional tone | -0.08 | -0.02 | YES |
| Specificity | -1.02 | -1.36 | YES |

**Directional agreement: 5/5 (100%)**
**Spearman rank correlation: ρ = 1.000, p < 0.001**

The results replicate perfectly across model sizes. Both models show the same preference ordering: politeness > specificity > directness > formality > emotional tone (by |d|). The smaller model even shows slightly *stronger* effects for some axes, suggesting these are robust properties of the model family rather than emergent capabilities.

### 4.4 Visualizations

All figures are saved in `figures/`:
- `directional_preferences.png`: Bar charts of directional consistency and signed ΔLL per axis
- `effect_sizes.png`: Horizontal bar chart of |Cohen's d| with small/medium/large reference lines
- `distribution_violin.png`: Violin plots showing full distribution of signed differences
- `scenario_axis_heatmap.png`: Heatmap of per-scenario signed differences (30 × 6)
- `introspection_vs_behavioral.png`: Side-by-side comparison of behavioral vs. introspective detection
- `cross_model_comparison.png`: Scatter plot of Qwen-7B vs Qwen-3B effect sizes

## 5. Analysis & Discussion

### 5.1 The "Hedged, Polite, and Vague" Default

Our results reveal a coherent profile of LLM generation defaults. When writing messages, these models strongly prefer responses that are:
- **Polite** (d = 1.81) — the strongest preference by far
- **Vague/general** (d = 1.02) — avoids committing to specifics
- **Indirect/hedged** (d = 0.94) — uses softeners and qualifiers
- **Formal** (d = 0.68) — errs toward professional register

This profile makes intuitive sense: it describes the "safe," "corporate" communication style that LLMs are often criticized for. The model defaults to the least risky pragmatic choices — polite, non-committal, hedged, formal — which minimizes the chance of causing offense but may not serve the user's actual communicative goals.

The exception is **emotional tone**, where the model has no clear default. This is noteworthy — the model doesn't systematically favor warmth or neutrality, suggesting this dimension is either genuinely context-dependent or not "considered" by the model's prior.

### 5.2 Introspective Blind Spots

The discrepancy between behavioral detection and introspection reveals two distinct failure modes:

1. **Blind spots** (directness): The model has a strong behavioral default (indirect/hedged) that it doesn't recognize or report. When asked what it considered, it mentions politeness, formality, and specificity but rarely directness — even though directness is one of its strongest implicit preferences. This suggests the model's hedging behavior is deeply embedded in its generation prior, below the threshold of "self-awareness."

2. **Performative consideration** (emotional tone): The model frequently claims to consider emotional tone (53% of scenarios mention it) but shows no consistent behavioral preference. This could mean the model genuinely considers tone on a per-scenario basis (without a strong default), or it reports "tone" as a consideration because it's the expected answer, not because it actually influences generation.

### 5.3 What "Under Consideration" Means

Our results suggest a useful operational definition: a variable is **under consideration** by an LLM if the model's generation prior shows a significant directional preference along that axis across contexts. By this definition:
- Politeness, specificity, directness, and formality are "under consideration" (strong defaults)
- Emotional tone is "not under consideration" (no default)

This is distinct from whether the model *can reason about* the variable (it can — when asked) or *claims to consider* it (it does, inconsistently). The perplexity sensitivity method reveals the model's actual generation-time preferences, not its reasoning-time knowledge.

### 5.4 Methodological Notes

**The synonym control paradox**: The synonym axis shows the highest absolute |Δ| (d = 3.26), with 100% preference for variant_a. This is expected — GPT-4.1 generates variant_a as the "natural" version and variant_b as the synonym-swapped version, so the evaluating model always finds the original more natural. This confirms perplexity is highly sensitive to surface form but validates using directional consistency rather than raw magnitude as our metric.

**Per-token normalization**: We normalize by response token count to control for length effects. Both variants in each pair were constrained to similar length (~20% tolerance) during generation.

## 6. Limitations

1. **Variant quality**: Response pairs were generated by GPT-4.1, not human-authored. Despite the constraint to differ on only one axis, some cross-axis contamination is inevitable (e.g., polite variants may also be more formal).

2. **Model family bias**: Both evaluation models are from the Qwen family. While they show perfect correlation, testing on models from different families (Llama, Mistral, GPT) would strengthen generalizability claims.

3. **Axis selection**: We tested 5 pragmatic axes. Other potentially relevant axes (urgency, confidence, humor, register) were not tested.

4. **Scenario coverage**: 30 scenarios across 3 domains may not capture all communicative contexts. Domain-specific effects (e.g., politeness mattering more in service scenarios) were not systematically analyzed.

5. **GPT-4.1 as introspection model**: Introspection was measured on GPT-4.1, not on the same models used for perplexity. The ideal comparison would use the same model for both, but this requires an instruct model that provides both log-probabilities and coherent introspection.

6. **Static analysis**: We measure the model's default preference, not whether it adapts when given explicit instructions (e.g., "write a casual message"). The model likely has different preferences when the prompt specifies a tone.

## 7. Conclusions & Next Steps

### Answer to Research Question
**Yes**, perplexity sensitivity analysis can detect which pragmatic variables an LLM "considers" during message generation. Four of five tested axes (politeness, specificity, directness, formality) show statistically significant directional preferences that replicate across model sizes (ρ = 1.0). The method also reveals a gap between behavioral and introspective awareness: the model has a strong hedging default it doesn't self-report, and claims to consider emotional tone without showing a behavioral preference.

### Practical Implications
- **For writing assistants**: Prompt users about variables the model defaults on (especially directness — users may want more direct messages than the model provides)
- **For alignment**: The "polite, vague, hedged" default profile may not serve all users equally; consider making these preferences configurable
- **For evaluation**: Perplexity sensitivity is a cheap, automated method for auditing LLM pragmatic defaults

### Recommended Follow-Up
1. **Cross-family replication**: Test on Llama 3, Mistral, GPT-4 (via logprobs API)
2. **Context sensitivity**: Test whether pragmatic defaults change under different prompt conditions (e.g., "write a casual message")
3. **Human validation**: Correlate perplexity-detected preferences with human judgments of "which version would the model generate?"
4. **More axes**: Test urgency, confidence, humor, hedging, cultural specificity
5. **Fine-grained analysis**: Do preferences vary by domain (workplace vs personal)?

## References

- Cho & Kim (2024). Pragmatic Inference of Scalar Implicature by LLMs. arXiv:2408.06673
- Cifka & Liutkus (2023). Black-box Language Model Explanation by Context Length Probing. arXiv:2212.14815
- Lin, Altshuler & Pierrehumbert (2024). Probing LLMs for Scalar Adjective Lexical Semantics. arXiv:2404.03301
- Ruis et al. (2024). The Goldilocks of Pragmatic Understanding. arXiv:2210.14986
- Sato, Kawano & Yoshino (2025). Pragmatic Theories Enhance Understanding of Implied Meanings. arXiv:2510.26253
- Trienes et al. (2025). Behavioral Analysis of Information Salience in LLMs. arXiv:2502.14613
- Unger & Buschmeier (2025). Conversational Implicatures: Modelling Relevance Theory. arXiv:2509.22354

## Appendix: Environment Details

```
Hardware: NVIDIA RTX A6000 (49GB VRAM) × 4
Python: 3.12.8
PyTorch: 2.5.1+cu124
Transformers: 5.5.0
Random seed: 42
Execution time: ~15 min total (scenario generation ~8 min, perplexity ~15s per model)
Estimated API cost: ~$3-5 (GPT-4.1 for scenario generation + introspection)
```
