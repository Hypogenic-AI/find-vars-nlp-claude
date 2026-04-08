"""
Measure per-token log-likelihood (perplexity proxy) of pre-filled response variants
under local LLMs using forced decoding.

For each (scenario_prompt, response_variant) pair, we compute the conditional
log-likelihood of the response given the prompt. This tells us how "natural"
the model finds each variant as a continuation.
"""

import json
import os
import sys
import time
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_model(model_name, device="cuda:0"):
    """Load a causal LM and tokenizer."""
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


def compute_response_loglikelihood(model, tokenizer, prompt, response, device="cuda:0"):
    """
    Compute the per-token log-likelihood of `response` conditioned on `prompt`.

    We concatenate the chat-formatted prompt + response, tokenize, run forward pass,
    and extract log-probs only for the response tokens.

    Returns:
        total_loglik: sum of log-probs for response tokens
        mean_loglik: mean log-prob per response token
        n_tokens: number of response tokens
    """
    # Format as chat message
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    # Try to use chat template; fall back to simple concatenation
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        full_text = f"User: {prompt}\nAssistant: {response}"
        prompt_only = f"User: {prompt}\nAssistant: "

    # Tokenize
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    prompt_ids = tokenizer.encode(prompt_only, return_tensors="pt").to(device)

    prompt_len = prompt_ids.shape[1]
    total_len = full_ids.shape[1]
    response_len = total_len - prompt_len

    if response_len <= 0:
        return 0.0, 0.0, 0

    # Forward pass — no gradient needed
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute log-probs for response tokens
    # logits[0, t, :] predicts token at position t+1
    # So for response tokens at positions [prompt_len, total_len-1],
    # we use logits at positions [prompt_len-1, total_len-2]
    response_logits = logits[0, prompt_len - 1 : total_len - 1, :]  # (response_len, vocab_size)
    response_token_ids = full_ids[0, prompt_len:total_len]  # (response_len,)

    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs[torch.arange(response_len), response_token_ids]

    total_loglik = token_log_probs.sum().item()
    mean_loglik = token_log_probs.mean().item()

    return total_loglik, mean_loglik, response_len


def process_variants(model, tokenizer, scenarios, variants, device="cuda:0"):
    """Compute perplexity for all variant pairs."""
    # Build scenario lookup
    scenario_map = {s["id"]: s for s in scenarios}

    results = []
    for i, variant in enumerate(variants):
        sid = variant["scenario_id"]
        scenario = scenario_map.get(sid)
        if not scenario:
            continue

        prompt = scenario["prompt"]
        axis = variant["axis"]

        # Measure variant_a
        total_a, mean_a, n_a = compute_response_loglikelihood(
            model, tokenizer, prompt, variant["variant_a"], device
        )
        # Measure variant_b
        total_b, mean_b, n_b = compute_response_loglikelihood(
            model, tokenizer, prompt, variant["variant_b"], device
        )

        result = {
            "scenario_id": sid,
            "axis": axis,
            "pole_a_label": variant.get("pole_a_label", "a"),
            "pole_b_label": variant.get("pole_b_label", "b"),
            "variant_a_mean_loglik": mean_a,
            "variant_b_mean_loglik": mean_b,
            "variant_a_total_loglik": total_a,
            "variant_b_total_loglik": total_b,
            "variant_a_n_tokens": n_a,
            "variant_b_n_tokens": n_b,
            "loglik_diff": mean_b - mean_a,  # positive = model prefers variant_b
            "abs_loglik_diff": abs(mean_b - mean_a),
        }
        results.append(result)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(variants)}] scenario={sid} axis={axis} "
                  f"diff={result['loglik_diff']:.4f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for output filename")
    args = parser.parse_args()

    # Load data
    with open("results/scenarios.json") as f:
        scenarios = json.load(f)
    with open("results/variants.json") as f:
        variants = json.load(f)

    print(f"Loaded {len(scenarios)} scenarios, {len(variants)} variants")

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Process
    print(f"\nComputing perplexities for {len(variants)} variants...")
    start = time.time()
    results = process_variants(model, tokenizer, scenarios, variants, args.device)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")

    # Save
    model_short = args.model.split("/")[-1]
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    outfile = f"results/model_outputs/perplexity_{model_short}{suffix}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {outfile}")

    # Quick summary
    print("\n=== Quick Summary ===")
    from collections import defaultdict
    axis_diffs = defaultdict(list)
    for r in results:
        axis_diffs[r["axis"]].append(r["abs_loglik_diff"])
    for axis, diffs in sorted(axis_diffs.items()):
        print(f"  {axis:20s}: mean |Δ| = {np.mean(diffs):.4f} ± {np.std(diffs):.4f}")


if __name__ == "__main__":
    main()
