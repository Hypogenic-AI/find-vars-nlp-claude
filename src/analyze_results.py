"""
Statistical analysis and visualization of perplexity sensitivity results.

Key metrics:
1. Directional consistency: Does the model consistently prefer one pole?
   (Sign test + Wilcoxon signed-rank on signed differences)
2. Sensitivity magnitude: How large are the perplexity differences?
3. Introspection comparison: Do self-reported variables match behavioral detection?
4. Cross-model consistency: Do different models show similar patterns?
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PRAGMATIC_AXES = ["formality", "directness", "politeness", "emotional_tone", "specificity"]
CONTROL_AXES = ["synonym"]
ALL_AXES = PRAGMATIC_AXES + CONTROL_AXES

# Map introspection variable names to our axes
INTROSPECTION_AXIS_MAP = {
    "formality": "formality", "formal": "formality", "register": "formality",
    "informal": "formality", "casual": "formality", "professional": "formality",
    "tone": "emotional_tone", "emotional tone": "emotional_tone",
    "emotion": "emotional_tone", "empathy": "emotional_tone",
    "warmth": "emotional_tone", "sentiment": "emotional_tone",
    "friendly": "emotional_tone", "compassion": "emotional_tone",
    "directness": "directness", "direct": "directness",
    "hedging": "directness", "indirectness": "directness",
    "assertiveness": "directness", "assertive": "directness",
    "politeness": "politeness", "polite": "politeness",
    "courtesy": "politeness", "respect": "politeness",
    "deference": "politeness", "respectful": "politeness",
    "specificity": "specificity", "specific": "specificity",
    "detail": "specificity", "concreteness": "specificity",
    "clarity": "specificity", "precision": "specificity",
    "concrete": "specificity", "vague": "specificity",
}

AXIS_POLE_LABELS = {
    "formality": ("casual", "formal"),
    "directness": ("indirect", "direct"),
    "politeness": ("blunt", "polite"),
    "emotional_tone": ("neutral", "warm"),
    "specificity": ("vague", "specific"),
    "synonym": ("version_a", "version_b"),
}


def load_perplexity_results(filepath):
    with open(filepath) as f:
        return pd.DataFrame(json.load(f))


def compute_statistics(df):
    """Compute per-axis statistics with focus on directional consistency."""
    results = {}
    bonferroni_k = len(ALL_AXES)

    for axis in ALL_AXES:
        axis_data = df[df["axis"] == axis]
        if len(axis_data) == 0:
            continue

        signed_diffs = axis_data["loglik_diff"].values
        abs_diffs = axis_data["abs_loglik_diff"].values
        n = len(signed_diffs)

        # Directional consistency: fraction preferring pole_b (positive diff)
        n_prefer_b = sum(1 for d in signed_diffs if d > 0)
        pct_prefer_b = n_prefer_b / n

        # Sign test: is the directional preference significantly different from chance?
        sign_p = stats.binomtest(n_prefer_b, n, 0.5, alternative="two-sided").pvalue

        # Wilcoxon signed-rank test on signed differences
        if n >= 5:
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(signed_diffs, alternative="two-sided")
            except ValueError:
                wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        else:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan

        # Effect size: Cohen's d of signed differences from 0
        cohens_d_signed = np.mean(signed_diffs) / np.std(signed_diffs) if np.std(signed_diffs) > 0 else np.inf

        # Directional consistency strength (how far from 0.5)
        consistency = abs(pct_prefer_b - 0.5) * 2  # 0 = no consistency, 1 = perfect

        # Bootstrap 95% CI for mean signed difference
        boot_means = []
        rng = np.random.RandomState(42)
        for _ in range(10000):
            sample = rng.choice(signed_diffs, size=n, replace=True)
            boot_means.append(np.mean(sample))
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        results[axis] = {
            "n": n,
            "mean_signed_diff": float(np.mean(signed_diffs)),
            "std_signed_diff": float(np.std(signed_diffs)),
            "mean_abs_diff": float(np.mean(abs_diffs)),
            "median_abs_diff": float(np.median(abs_diffs)),
            "pct_prefer_b": float(pct_prefer_b),
            "preferred_pole": AXIS_POLE_LABELS.get(axis, ("a", "b"))[1] if pct_prefer_b > 0.5 else AXIS_POLE_LABELS.get(axis, ("a", "b"))[0],
            "directional_consistency": float(consistency),
            "sign_test_p": float(sign_p),
            "wilcoxon_p": float(wilcoxon_p),
            "cohens_d": float(cohens_d_signed),
            "abs_cohens_d": float(abs(cohens_d_signed)),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "is_pragmatic": axis in PRAGMATIC_AXES,
            "significant_sign": sign_p < 0.05 / bonferroni_k,
            "significant_wilcoxon": wilcoxon_p < 0.05 / bonferroni_k if not np.isnan(wilcoxon_p) else False,
        }

    return results


def analyze_introspection(introspections, axis_stats):
    """Compare introspection-reported variables with perplexity-detected ones."""
    axis_mention_counts = defaultdict(int)
    total_scenarios = len(introspections)

    for intro in introspections:
        mentioned_axes = set()
        for var_info in intro.get("variables_considered", []):
            var_name = var_info.get("variable", "").lower().strip()
            for keyword, axis in INTROSPECTION_AXIS_MAP.items():
                if keyword in var_name:
                    mentioned_axes.add(axis)
        for axis in mentioned_axes:
            axis_mention_counts[axis] += 1

    introspection_rates = {}
    for axis in PRAGMATIC_AXES:
        rate = axis_mention_counts.get(axis, 0) / max(total_scenarios, 1)
        stat = axis_stats.get(axis, {})
        introspection_rates[axis] = {
            "mention_rate": float(rate),
            "mention_count": int(axis_mention_counts.get(axis, 0)),
            "detected_by_perplexity": bool(stat.get("significant_wilcoxon", False)),
            "reported_by_introspection": rate > 0.3,
            "perplexity_consistency": float(stat.get("directional_consistency", 0)),
            "perplexity_preferred_pole": stat.get("preferred_pole", "unknown"),
        }

    return introspection_rates


def plot_directional_preferences(axis_stats, output_dir, model_name=""):
    """Bar chart showing directional preference strength per axis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    axes = PRAGMATIC_AXES + CONTROL_AXES
    labels = [a.replace("_", "\n") for a in axes]

    # Left: Directional consistency (distance from 50%)
    consistencies = [axis_stats.get(a, {}).get("directional_consistency", 0) for a in axes]
    pct_b = [axis_stats.get(a, {}).get("pct_prefer_b", 0.5) for a in axes]
    colors_left = []
    for a in axes:
        s = axis_stats.get(a, {})
        if s.get("significant_sign"):
            colors_left.append("#E53935" if a in PRAGMATIC_AXES else "#757575")
        else:
            colors_left.append("#90CAF9" if a in PRAGMATIC_AXES else "#BDBDBD")

    ax1.bar(labels, consistencies, color=colors_left, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Directional Consistency\n(0 = no preference, 1 = perfect)", fontsize=11)
    ax1.set_title(f"How Consistently Does the Model Prefer One Pole?\n{model_name}", fontsize=13)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    # Add preferred pole labels
    for i, a in enumerate(axes):
        s = axis_stats.get(a, {})
        pole = s.get("preferred_pole", "?")
        pref = s.get("pct_prefer_b", 0.5)
        pref_pct = max(pref, 1 - pref) * 100
        ax1.text(i, consistencies[i] + 0.02, f"{pole}\n({pref_pct:.0f}%)",
                ha="center", va="bottom", fontsize=8, style="italic")

    # Right: Signed mean log-likelihood difference with CI
    means = [axis_stats.get(a, {}).get("mean_signed_diff", 0) for a in axes]
    ci_lows = [axis_stats.get(a, {}).get("ci_95_low", 0) for a in axes]
    ci_highs = [axis_stats.get(a, {}).get("ci_95_high", 0) for a in axes]
    errors_low = [m - cl for m, cl in zip(means, ci_lows)]
    errors_high = [ch - m for m, ch in zip(means, ci_highs)]
    colors_right = ["#2196F3" if a in PRAGMATIC_AXES else "#9E9E9E" for a in axes]

    ax2.bar(labels, means, color=colors_right, edgecolor="black", linewidth=0.5)
    ax2.errorbar(labels, means, yerr=[errors_low, errors_high],
                fmt="none", color="black", capsize=5)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Mean Signed Δ Log-Likelihood\n(+ = prefers pole_b, - = prefers pole_a)", fontsize=11)
    ax2.set_title(f"Direction and Magnitude of Preference\n(with 95% Bootstrap CI)", fontsize=13)

    # Add pole labels to axis
    for i, a in enumerate(axes):
        pole_a, pole_b = AXIS_POLE_LABELS.get(a, ("a", "b"))
        if i == 0:
            ax2.annotate(f"← {pole_a}", xy=(i, min(means) * 0.95), fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "directional_preferences.png"), dpi=150)
    plt.close()
    print("Saved directional_preferences.png")


def plot_introspection_comparison(intro_rates, axis_stats, output_dir):
    """Compare perplexity-detected variables vs introspection-reported ones."""
    fig, ax = plt.subplots(figsize=(10, 7))

    axes = PRAGMATIC_AXES
    labels = [a.replace("_", "\n") for a in axes]

    # Perplexity consistency (behavioral)
    ppl_consistency = [axis_stats.get(a, {}).get("directional_consistency", 0) for a in axes]
    # Introspection mention rate
    intro_mention = [intro_rates.get(a, {}).get("mention_rate", 0) for a in axes]

    x = np.arange(len(axes))
    width = 0.35

    bars1 = ax.bar(x - width/2, ppl_consistency, width,
                   label="Perplexity Consistency\n(behavioral)", color="#2196F3", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, intro_mention, width,
                   label="Introspection Rate\n(self-reported)", color="#FF9800", edgecolor="black", linewidth=0.5)

    # Significance markers
    for i, a in enumerate(axes):
        s = axis_stats.get(a, {})
        if s.get("significant_sign"):
            ax.text(x[i] - width/2, ppl_consistency[i] + 0.02, "*", ha="center", fontsize=14, color="red")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Behavioral (Perplexity) vs. Introspective Variable Detection\n* = significant directional preference (Bonferroni-corrected)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "introspection_vs_behavioral.png"), dpi=150)
    plt.close()
    print("Saved introspection_vs_behavioral.png")


def plot_per_scenario_heatmap(df, output_dir):
    """Heatmap of signed perplexity differences (scenario × axis)."""
    pivot = df.pivot_table(values="loglik_diff", index="scenario_id", columns="axis", aggfunc="mean")
    col_order = [c for c in PRAGMATIC_AXES + CONTROL_AXES if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(12, 10))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt=".2f", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Δ Log-likelihood (+ = prefers pole_b)"})
    ax.set_title("Signed Perplexity Sensitivity: Scenario × Axis", fontsize=14)
    ax.set_xlabel("Pragmatic Axis")
    ax.set_ylabel("Scenario ID")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario_axis_heatmap.png"), dpi=150)
    plt.close()
    print("Saved scenario_axis_heatmap.png")


def plot_distribution(df, output_dir):
    """Violin plots of signed Δ log-likelihood per axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = PRAGMATIC_AXES + CONTROL_AXES
    plot_data = df[df["axis"].isin(order)].copy()

    sns.violinplot(data=plot_data, x="axis", y="loglik_diff", order=order,
                   inner="box", palette="Set2", ax=ax)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Axis", fontsize=12)
    ax.set_ylabel("Signed Δ Log-Likelihood per Token", fontsize=12)
    ax.set_title("Distribution of Perplexity Preferences per Axis\n(Above 0 = prefers pole_b, Below 0 = prefers pole_a)", fontsize=13)
    ax.set_xticklabels([a.replace("_", "\n") for a in order])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_violin.png"), dpi=150)
    plt.close()
    print("Saved distribution_violin.png")


def plot_effect_sizes(axis_stats, output_dir):
    """Horizontal bar chart of |Cohen's d| per axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    axes = PRAGMATIC_AXES + CONTROL_AXES
    ds = [axis_stats.get(a, {}).get("abs_cohens_d", 0) for a in axes]
    colors = ["#4CAF50" if a in PRAGMATIC_AXES else "#9E9E9E" for a in axes]
    labels = [a.replace("_", " ") for a in axes]

    ax.barh(labels, ds, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0.2, color="orange", linestyle="--", alpha=0.5, label="Small (0.2)")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Medium (0.5)")
    ax.axvline(x=0.8, color="darkred", linestyle="--", alpha=0.5, label="Large (0.8)")

    ax.set_xlabel("|Cohen's d| (signed difference from 0)", fontsize=12)
    ax.set_title("Effect Size of Directional Preference per Axis", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "effect_sizes.png"), dpi=150)
    plt.close()
    print("Saved effect_sizes.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--perplexity-file", type=str, required=True)
    parser.add_argument("--perplexity-file-2", type=str, default=None)
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    df = load_perplexity_results(args.perplexity_file)
    model_name = os.path.basename(args.perplexity_file).replace("perplexity_", "").replace(".json", "")
    print(f"Loaded {len(df)} results for {model_name}")

    introspections = []
    if os.path.exists("results/introspections.json"):
        with open("results/introspections.json") as f:
            introspections = json.load(f)

    # === STATISTICAL ANALYSIS ===
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    axis_stats = compute_statistics(df)
    bonferroni_alpha = 0.05 / len(ALL_AXES)

    print(f"\nBonferroni-corrected α = {bonferroni_alpha:.4f}")
    print(f"\n{'Axis':<18} {'Mean Δ':>9} {'%→pole_b':>10} {'Preferred':>12} {'Consist.':>10} "
          f"{'Sign p':>10} {'Wilcox p':>10} {'Cohens d':>10} {'Sig?':>6}")
    print("-" * 105)
    for axis in PRAGMATIC_AXES + CONTROL_AXES:
        s = axis_stats.get(axis, {})
        if not s:
            continue
        sig = ""
        if s.get("significant_sign") and s.get("significant_wilcoxon"):
            sig = "BOTH"
        elif s.get("significant_sign"):
            sig = "sign"
        elif s.get("significant_wilcoxon"):
            sig = "wilc"
        else:
            sig = "ns"
        print(f"{axis:<18} {s['mean_signed_diff']:>+9.4f} {s['pct_prefer_b']*100:>9.1f}% "
              f"{s['preferred_pole']:>12} {s['directional_consistency']:>10.3f} "
              f"{s['sign_test_p']:>10.6f} {s['wilcoxon_p']:>10.6f} {s['cohens_d']:>+10.2f} {sig:>6}")

    # === INTROSPECTION ANALYSIS ===
    if introspections:
        print("\n" + "=" * 80)
        print("INTROSPECTION vs BEHAVIORAL ANALYSIS")
        print("=" * 80)
        intro_rates = analyze_introspection(introspections, axis_stats)

        print(f"\n{'Axis':<18} {'Intro %':>10} {'PPL Consist':>12} {'PPL pref':>12} "
              f"{'PPL sig?':>10} {'Intro rept?':>12} {'Match?':>8}")
        print("-" * 86)
        for axis in PRAGMATIC_AXES:
            r = intro_rates.get(axis, {})
            ppl_sig = "YES" if r.get("detected_by_perplexity") else "no"
            intro_rep = "YES" if r.get("reported_by_introspection") else "no"
            agree = "AGREE" if (r.get("detected_by_perplexity") == r.get("reported_by_introspection")) else "DIFF"
            print(f"{axis:<18} {r.get('mention_rate', 0)*100:>9.1f}% "
                  f"{r.get('perplexity_consistency', 0):>12.3f} "
                  f"{r.get('perplexity_preferred_pole', '?'):>12} "
                  f"{ppl_sig:>10} {intro_rep:>12} {agree:>8}")

        # Summary of agreement
        n_agree = sum(1 for a in PRAGMATIC_AXES
                     if intro_rates.get(a, {}).get("detected_by_perplexity") ==
                        intro_rates.get(a, {}).get("reported_by_introspection"))
        print(f"\nAgreement: {n_agree}/{len(PRAGMATIC_AXES)} axes ({n_agree/len(PRAGMATIC_AXES)*100:.0f}%)")
    else:
        intro_rates = {}

    # === VISUALIZATIONS ===
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_directional_preferences(axis_stats, "figures", model_name)
    plot_effect_sizes(axis_stats, "figures")
    plot_distribution(df, "figures")
    plot_per_scenario_heatmap(df, "figures")
    if intro_rates:
        plot_introspection_comparison(intro_rates, axis_stats, "figures")

    # === CROSS-MODEL COMPARISON ===
    if args.perplexity_file_2 and os.path.exists(args.perplexity_file_2):
        print("\n" + "=" * 80)
        print("CROSS-MODEL COMPARISON")
        print("=" * 80)
        df2 = load_perplexity_results(args.perplexity_file_2)
        model2_name = os.path.basename(args.perplexity_file_2).replace("perplexity_", "").replace(".json", "")
        axis_stats2 = compute_statistics(df2)

        print(f"\n{'Axis':<18} {model_name+' d':>15} {model2_name+' d':>15} {'Same dir?':>10}")
        print("-" * 62)

        m1_ds, m2_ds = [], []
        same_dir_count = 0
        for axis in PRAGMATIC_AXES:
            s1 = axis_stats.get(axis, {})
            s2 = axis_stats2.get(axis, {})
            d1 = s1.get("cohens_d", 0)
            d2 = s2.get("cohens_d", 0)
            m1_ds.append(d1)
            m2_ds.append(d2)
            same = "YES" if (d1 > 0) == (d2 > 0) else "NO"
            if same == "YES":
                same_dir_count += 1
            print(f"{axis:<18} {d1:>+15.3f} {d2:>+15.3f} {same:>10}")

        print(f"\nDirectional agreement: {same_dir_count}/{len(PRAGMATIC_AXES)}")

        if len(m1_ds) >= 3:
            corr, p_corr = stats.spearmanr(m1_ds, m2_ds)
            print(f"Spearman rank correlation of Cohen's d: ρ = {corr:.3f}, p = {p_corr:.4f}")

        # Cross-model scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(m1_ds, m2_ds, s=120, color="#2196F3", edgecolor="black", zorder=5)
        for i, axis in enumerate(PRAGMATIC_AXES):
            ax.annotate(axis.replace("_", "\n"), (m1_ds[i], m2_ds[i]),
                       textcoords="offset points", xytext=(8, 5), fontsize=10)
        lims = [min(min(m1_ds), min(m2_ds)) - 0.2, max(max(m1_ds), max(m2_ds)) + 0.2]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5)
        ax.axhline(0, color="gray", alpha=0.3)
        ax.axvline(0, color="gray", alpha=0.3)
        ax.set_xlabel(f"{model_name} Cohen's d", fontsize=12)
        ax.set_ylabel(f"{model2_name} Cohen's d", fontsize=12)
        ax.set_title("Cross-Model Consistency of Directional Preferences", fontsize=14)
        plt.tight_layout()
        plt.savefig("figures/cross_model_comparison.png", dpi=150)
        plt.close()
        print("Saved cross_model_comparison.png")

    # Save all statistics
    all_stats = {
        "model": model_name,
        "bonferroni_alpha": bonferroni_alpha,
        "axis_statistics": axis_stats,
        "introspection_rates": {k: v for k, v in intro_rates.items()} if intro_rates else {},
    }
    with open("results/analysis_statistics.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nSaved analysis_statistics.json")


if __name__ == "__main__":
    main()
