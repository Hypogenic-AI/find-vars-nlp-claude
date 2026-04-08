# Research Plan: Finding Variables Under Consideration

## Motivation & Novelty Assessment

### Why This Research Matters
When LLMs help users write messages, they make implicit choices about pragmatic variables (formality, directness, politeness, emotional tone) without explicitly reasoning about them. Yet when asked, LLMs can readily identify and discuss these pragmatic dimensions. This gap between *latent knowledge* and *active consideration during generation* has practical implications: if we can detect which variables a model is actively "considering," we can improve generation quality, build better human-AI writing tools, and understand LLM pragmatic competence more deeply.

### Gap in Existing Work
The literature review reveals three key gaps:
1. **Production vs. comprehension**: Existing probing work (Lin et al. 2024, Cho & Kim 2024) focuses on whether LLMs *understand* pragmatic phenomena, not on what they *actively consider* during generation.
2. **Pre-filling as a probing tool**: While forced decoding is used in evaluation, no one has systematically used pre-filled response variants to probe which axes of variation the model is sensitive to.
3. **Cross-axis comparison**: No unified framework compares model sensitivity across multiple pragmatic dimensions simultaneously.

Trienes et al. (2025) showed that LLM self-reported salience diverges from behavioral salience — our method provides a new behavioral probe via perplexity sensitivity.

### Our Novel Contribution
We propose **perplexity sensitivity analysis** as a method to detect "variables under consideration" by an LLM. Given a message-writing prompt, we:
1. Generate response variants that systematically differ along pragmatic axes
2. Measure log-likelihood (perplexity) of each variant under the model
3. Test whether perplexity is significantly sensitive to each axis
4. Compare behavioral sensitivity (perplexity) against introspective reports (what the model says it considered)

A variable is "under consideration" if the model assigns meaningfully different likelihoods to responses that differ along that axis (controlling for surface-level changes).

### Experiment Justification
- **Experiment 1 (Perplexity Sensitivity)**: Core test — do models show differential perplexity across pragmatic axis variations? This directly tests the hypothesis.
- **Experiment 2 (Introspection Comparison)**: Tests whether self-reported variables match perplexity-detected variables, validating the behavioral probing approach.
- **Experiment 3 (Cross-Model Comparison)**: Tests generalizability across model families.

## Research Question
Can we identify which pragmatic variables an LLM "considers" during message generation by measuring whether systematically varying those variables in pre-filled responses produces significant perplexity changes?

## Hypothesis Decomposition
- **H1**: Pre-filled responses varying along pragmatic axes (formality, directness, politeness, emotion, specificity) will show significantly different perplexities under the generating model.
- **H2**: Not all axes will show equal sensitivity — some variables are more "under consideration" than others.
- **H3**: The set of variables detected via perplexity sensitivity will differ from variables the model reports when asked directly (introspection ≠ behavior).
- **H4**: Perplexity sensitivity patterns will be partially consistent across different model families.

## Proposed Methodology

### Approach
We use **perplexity sensitivity analysis** — a black-box behavioral probing method. For each scenario, we construct response pairs that differ along one pragmatic axis while holding others constant. We then compute the log-likelihood of each response under the model and test whether the axis-specific difference is statistically significant across scenarios.

### Pragmatic Axes to Test
1. **Formality**: casual/conversational ↔ formal/professional
2. **Directness**: hedged/indirect ↔ direct/assertive
3. **Politeness**: blunt/terse ↔ polite/courteous
4. **Emotional tone**: neutral/factual ↔ warm/empathetic
5. **Specificity**: vague/general ↔ detailed/specific

### Control Axes
6. **Synonym substitution**: Replace words with synonyms (surface change, no pragmatic change)
7. **Length padding**: Add filler phrases (controls for length effects)

### Experimental Steps

1. **Scenario Construction** (using GPT-4.1):
   - Generate 30 diverse message-writing scenarios across domains:
     - Workplace (emails to boss, colleagues, clients)
     - Personal (messages to friends, family, acquaintances)
     - Service (complaints, requests, feedback)
   - Each scenario = a user prompt asking the LLM to write a message

2. **Baseline Response Generation**:
   - For each scenario, generate a default response from the local model
   - This serves as the "natural" response

3. **Axis Variant Generation** (using GPT-4.1):
   - For each scenario, generate response pairs along each axis:
     - High-formality vs. low-formality version
     - Direct vs. indirect version
     - etc.
   - Constraint: variants should differ primarily on the target axis
   - Also generate control variants (synonym swap, length padding)

4. **Perplexity Computation** (local model on GPU):
   - For each (scenario, variant) pair, compute the conditional log-likelihood of the response given the prompt
   - Use token-level log-probabilities from the model
   - Normalize by response length (per-token log-likelihood)

5. **Introspection Probing** (using GPT-4.1):
   - For each scenario, ask the model: "What pragmatic variables did you consider when writing this message?"
   - Code the self-reported variables

6. **Statistical Analysis**:
   - For each axis: paired t-test (or Wilcoxon) comparing perplexity of high vs. low variants across scenarios
   - Effect sizes (Cohen's d) for each axis
   - Comparison of perplexity-sensitive axes vs. introspection-reported axes

### Models
- **Primary**: Qwen/Qwen2.5-7B-Instruct (local, GPU) — for perplexity measurement
- **Secondary**: microsoft/Phi-3.5-mini-instruct (local, GPU) — cross-model comparison
- **Scenario generation**: GPT-4.1 via OpenAI API

### Baselines
1. Control axes (synonym, length) should show minimal perplexity sensitivity
2. Random text modifications as noise baseline
3. Compare effect sizes across pragmatic vs. control axes

### Evaluation Metrics
- **Per-token log-likelihood** (negative = perplexity proxy)
- **Perplexity ratio**: PPL(variant_A) / PPL(variant_B) for each axis
- **Effect size** (Cohen's d) per axis across scenarios
- **Statistical significance** (Wilcoxon signed-rank test, corrected for multiple comparisons)
- **Introspection agreement**: % of perplexity-detected axes also self-reported

### Statistical Analysis Plan
- Wilcoxon signed-rank test for each axis (non-parametric, handles non-normal perplexity distributions)
- Bonferroni correction for 5 pragmatic axes + 2 controls = 7 comparisons
- α = 0.05 (corrected: α = 0.007)
- Report effect sizes and 95% bootstrap CIs

## Expected Outcomes
- **Supporting H1**: Pragmatic axes show statistically significant perplexity differences (p < 0.007 after correction)
- **Supporting H2**: Effect sizes vary across axes (e.g., formality may show stronger sensitivity than specificity)
- **Supporting H3**: Introspection mentions variables (e.g., "tone") that don't show perplexity sensitivity, and misses variables that do
- **Refuting hypothesis**: If control axes show similar effect sizes to pragmatic axes, or if no axis shows significant perplexity sensitivity

## Timeline and Milestones
1. Environment setup + dependency installation: 10 min
2. Scenario + variant generation via GPT-4.1: 20 min
3. Local model setup + perplexity computation: 30 min
4. Introspection probing: 10 min
5. Statistical analysis + visualization: 20 min
6. Documentation: 20 min

## Potential Challenges
- **Variant quality**: GPT-4.1-generated variants may differ on multiple axes simultaneously → mitigate with manual inspection and quality filtering
- **Length confound**: Different-length responses have different perplexities → normalize per-token
- **Model tokenization**: Token boundaries may differ across models → report both per-token and per-character metrics
- **Small N**: 30 scenarios × 7 axes may have limited statistical power → use non-parametric tests and report effect sizes

## Success Criteria
1. At least one pragmatic axis shows significantly larger perplexity sensitivity than control axes
2. Perplexity-detected variables partially diverge from introspection-reported variables
3. Results replicate across at least 2 model families
