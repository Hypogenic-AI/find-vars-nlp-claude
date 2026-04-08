# Finding Variables Under Consideration

Probing which pragmatic variables LLMs actually "consider" during message generation using perplexity sensitivity analysis.

## Key Findings

- **LLMs have a strong "safe default" profile**: polite (d=1.81), vague (d=1.02), indirect (d=0.94), and formal (d=0.68) — all statistically significant (p < 0.001)
- **Emotional tone is NOT actively considered**: no directional preference detected (d=-0.08, p=0.46)
- **Introspection != behavior**: The model doesn't report its strongest behavioral default (indirectness), and claims to consider a variable (emotional tone) with no behavioral signal. Agreement: 60%
- **Results replicate perfectly across model sizes**: Qwen-7B and Qwen-3B show identical preference ordering (Spearman rho = 1.0)
- **Method is cheap and fast**: ~15s of GPU time per model to evaluate 180 variant pairs

## Method

1. Generate 30 message-writing scenarios across workplace/personal/service domains
2. For each scenario, create response pairs varying along 5 pragmatic axes + 1 control
3. Measure per-token log-likelihood of each variant using a local model
4. Test for directional consistency (sign test + Wilcoxon signed-rank)
5. Compare behavioral detection against model introspection

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers accelerate openai scipy matplotlib numpy pandas seaborn

# Run experiments
export OPENAI_API_KEY=your_key_here
python src/generate_scenarios.py          # Generate scenarios + variants via GPT-4.1
python src/measure_perplexity.py          # Compute perplexity on local model (requires GPU)
python src/measure_perplexity.py --model Qwen/Qwen2.5-3B-Instruct --device cuda:1  # Second model
python src/analyze_results.py \
  --perplexity-file results/model_outputs/perplexity_Qwen2.5-7B-Instruct.json \
  --perplexity-file-2 results/model_outputs/perplexity_Qwen2.5-3B-Instruct.json
```

## File Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Experimental design and methodology plan
├── src/
│   ├── generate_scenarios.py  # Scenario + variant generation via GPT-4.1
│   ├── measure_perplexity.py  # Log-likelihood computation with local models
│   └── analyze_results.py     # Statistical analysis + visualization
├── results/
│   ├── scenarios.json         # 30 message-writing scenarios
│   ├── variants.json          # 180 axis-varied response pairs
│   ├── introspections.json    # GPT-4.1 self-reported variable lists
│   ├── analysis_statistics.json
│   └── model_outputs/         # Per-model perplexity results
├── figures/                   # All visualizations
├── literature_review.md       # Background literature synthesis
├── resources.md               # Dataset and code catalog
├── papers/                    # Downloaded research papers
├── datasets/                  # Downloaded datasets
└── code/                      # Cloned reference implementations
```

See [REPORT.md](REPORT.md) for full details.
