import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import jensenshannon
import json
import numpy as np

# Print JSD
def print_jsd_by_period(js_results):
    for period, result in js_results.items():
        print(f"\n=== Shift to period {period} ===")
        print(f"Jensen-Shannon Divergence: {result['JSD']:.4f}")
        print("Top shifted items:")
        for slot, score in result['top_shifted_items'].items():
            print(f"  {slot}: {score:.4f}")

# Plot JSD
def plot_jsd_by_period(js_results):
    periods = list(js_results.keys())
    jsd_scores = [js_results[d]['JSD'] for d in periods]

    plt.figure(figsize=(8, 5))
    plt.plot(periods, jsd_scores, marker='o')
    plt.title("Jensen-Shannon Divergence Between Periods")
    plt.xlabel("Decade")
    plt.ylabel("JSD (χ² distance)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot top-N shifting items
def plot_items_jsd_by_period(js_results, top_n=10, cols=3):
    """
    Plot all top-N shifting items across periods in a grid of subplots,
    with shared x-axis scaling across all plots.
    """
    num_periods = len(js_results)
    rows = math.ceil(num_periods / cols)

    # Find global max contribution across all periods
    global_max = max(
        result['top_shifted_items'].head(top_n).max()
        for result in js_results.values()
    )

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for idx, (decade, result) in enumerate(js_results.items()):
        ax = axes[idx]
        top_words = result['top_shifted_items'].head(top_n)

        # Colors by prefix
        colors = [
            "green" if w.startswith("in_") else 
            "red" if w.startswith("de_") else 
            "gray"
            for w in top_words.index
        ]

        sns.barplot(
            x=top_words.values,
            y=top_words.index.str.replace("in_", "", regex=False).str.replace("de_", "", regex=False),
            ax=ax,
            legend=False,
            hue=top_words.index,
            palette=dict(zip(top_words.index, colors))
        )

        ax.set_title(f"{decade} (JSD: {result['JSD']:.3f})", fontsize=10)
        ax.set_xlabel("JSD Contribution", fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)

        # Fix x-axis across all subplots
        ax.set_xlim(0, global_max * 1.05)  # small margin

    # Remove unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Add direction prefix for JSD visualisation
def direction_prefix_map(vocab, p, q, prefix_in="in_", prefix_de="de_", neutral=""):
    """
    Build a mapping {slot -> prefixed_slot} based on probability change p - q.
    p, q: 1D numpy arrays aligned with 'vocab' (same order, same length).
    """
    out = {}
    for i, slot in enumerate(vocab):
        if p[i] > q[i]:
            out[slot] = f"{prefix_in}{slot}"
        elif p[i] < q[i]:
            out[slot] = f"{prefix_de}{slot}"
        else:
            out[slot] = f"{neutral}{slot}"
    return out


# Compute JSD of syntactic slots across periods
def slots_jsd_by_period(json_path, top_n=10, min_count=0):
    """
    Compute JSD shift in the distribution of syntactic slots across periods.

    Parameters:
        json_path (str): Path to JSON file (period → slot counts).
        top_n (int): Use union of top-n slots from each period.
        min_count (int): Minimum combined count across both periods to keep a slot.

    Returns:
        dict: {period2: {'JSD': float, 'top_shifted_items': pd.Series}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='index').fillna(0).astype(int)
    periods = sorted(df.index)

    # Union of top-n slots across all periods
    slot_union = set()
    for p in periods:
        top_slots = df.loc[p].sort_values(ascending=False).head(top_n).index
        slot_union.update(top_slots)
    slot_union = sorted(slot_union)

    output = {}
    for i in range(1, len(periods)):
        p1, p2 = periods[i - 1], periods[i]
        v1 = df.loc[p1][slot_union]
        v2 = df.loc[p2][slot_union]

        # Filter by min count
        vocab = [s for s in slot_union if v1.get(s, 0) + v2.get(s, 0) >= min_count]
        q = np.array([v1.get(s, 0) for s in vocab], dtype=float)
        p = np.array([v2.get(s, 0) for s in vocab], dtype=float)

        if p.sum() == 0 or q.sum() == 0:
            continue

        p /= p.sum()
        q /= q.sum()

        jsd = jensenshannon(p, q, base=2) ** 2
        m = 0.5 * (p + q)
        pointwise_jsd = 0.5 * (p * np.log2(p / m + 1e-12) + q * np.log2(q / m + 1e-12))
        contrib = pd.Series(pointwise_jsd, index=vocab).sort_values(ascending=False)

        # Build prefixed names based on direction (increase/decrease/no change)
        name_map = direction_prefix_map(vocab, p, q, prefix_in="in_", prefix_de="de_", neutral="")
        contrib.index = [name_map[s] for s in contrib.index]

        output[p2] = {
            'JSD': jsd,
            'top_shifted_items': contrib[contrib > 0].head(10)
        }

    return output

def sfillers_jsd_by_period(df, word_col='chi_amod', period_col='half_decade', min_count=0):
    output = {}
    periods = sorted(df[period_col].dropna().unique())

    for i in range(1, len(periods)):
        d1, d2 = periods[i - 1], periods[i]
        f1 = df[df[period_col] == d1][word_col].value_counts()
        f2 = df[df[period_col] == d2][word_col].value_counts()

        vocab = sorted(set(f1.index) | set(f2.index))
        vocab = [w for w in vocab if f1.get(w, 0) + f2.get(w, 0) >= min_count]

        p = np.array([f2.get(w, 0) for w in vocab], dtype=float)
        q = np.array([f1.get(w, 0) for w in vocab], dtype=float)

        if p.sum() == 0 or q.sum() == 0:
            continue

        p /= p.sum()
        q /= q.sum()

        jsd = jensenshannon(p, q, base=2) ** 2

        m = 0.5 * (p + q)
        pointwise_jsd = 0.5 * (p * np.log2(p / m + 1e-12) + q * np.log2(q / m + 1e-12))
        contrib = pd.Series(pointwise_jsd, index=vocab).sort_values(ascending=False)

        # Build prefixed names based on direction (increase/decrease/no change)
        name_map = direction_prefix_map(vocab, p, q, prefix_in="in_", prefix_de="de_", neutral="")
        contrib.index = [name_map[s] for s in contrib.index]

        output[d2] = {
            'JSD': jsd,
            'top_shifted_items': contrib[contrib > 0].head(10)
        }

    return output