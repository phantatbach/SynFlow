import ast
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import seaborn as sns
import json
from typing import Dict, Tuple

#---------------------------------------------------------------
# Helper functions
def cal_jsd(distribution_1, distribution_2):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.

    Parameters:
        distribution_1 (numpy array): First probability distribution.
        distribution_2 (numpy array): Second probability distribution.

    Returns:
        float: The Jensen-Shannon divergence between p and q.
    """
    return jensenshannon(distribution_1, distribution_2, base=2)**2  # squared distance = divergence

def cal_contrib_jsd(distribution_1, distribution_2, vocab):
    """
    Decompose the global JSD score to individual items, computing the pointwise JSD
    between two probability distributions.

    Parameters:
        distribution_1 (numpy array): First probability distribution.
        distribution_2 (numpy array): Second probability distribution.
        vocab (list): List of slot types corresponding to the two distributions.

    Returns:
        list[dict]: Sorted list of dictionaries with keys ``item`` and ``contribution``.
    """
    distribution_mix = 0.5 * (distribution_1 + distribution_2)
    pointwise_jsd = 0.5 * (distribution_1 * np.log2(distribution_1 / distribution_mix + 1e-12) + 
                           distribution_2 * np.log2(distribution_2 / distribution_mix + 1e-12))

    name_map = direction_prefix_map(
        vocab,
        distribution_1,
        distribution_2,
        prefix_in="in_",
        prefix_de="de_",
        prefix_born="bo_",
        prefix_lost="lo_",
        neutral=""
    )
    contrib = [
        {"item": name_map[slot], "contribution": float(score)}
        for slot, score in sorted(
            zip(vocab, pointwise_jsd), key=lambda pair: pair[1], reverse=True
        )
    ]

    return contrib

# Add direction prefix for JSD visualisation
def direction_prefix_map(vocab, distribution_1, distribution_2, prefix_in="in_", prefix_de="de_",
                          prefix_born = 'bo_', prefix_lost = 'lo_', neutral=""):
    """
    Maps slot types to prefixed names based on the direction of the change.

    Parameters:
        vocab (list): List of slot types.
        distribution_1 (numpy array): First probability distribution.
        distribution_2 (numpy array): Second probability distribution.
        prefix_in (str): Prefix for slot types that have increased in frequency.
        prefix_de (str): Prefix for slot types that have decreased in frequency.
        neutral (str): Prefix for slot types that have not changed in frequency.

    Returns:
        dict: A dictionary with slot types as keys and prefixed names as values.
    """
    out = {}
    for i, slot in enumerate(vocab):
        if distribution_1[i] == 0 and distribution_2[i] > 0:
            out[slot] = f"{prefix_born}{slot}"
        elif distribution_1[i] > 0 and distribution_2[i] == 0:
            out[slot] = f"{prefix_lost}{slot}"
        elif distribution_1[i] == distribution_2[i]:
            out[slot] = f"{neutral}{slot}"
        elif distribution_1[i] > 0 and distribution_2[i] > 0:
            if distribution_2[i] > distribution_1[i]:
                out[slot] = f"{prefix_in}{slot}"
            elif distribution_2[i] < distribution_1[i]:
                out[slot] = f"{prefix_de}{slot}"
    return out

#---------------------------------------------------------------
# Print JSD
def print_jsd_by_period(js_results):
    """
    Print the Jensen-Shannon Divergence and top shifted items for each period.

    Parameters:
        js_results (dict): A dictionary with period as key and a dictionary as value.
            The dictionary contains the JSD and top shifted items.

    Returns:
        None
    """
    for period, result in js_results.items():
        print(f"\n=== Shift to period {period} ===")
        print(f"Jensen-Shannon Divergence: {result['JSD']:.4f}")
        print("Top shifted items:")
        for item in result['top_shifted_items']:
            print(f"  {item['item']}: {item['contribution']:.4f}")

# Plot JSD
def plot_jsd_by_period(js_results):
    """
    Plot the Jensen-Shannon Divergence between two periods.

    Parameters:
        js_results (dict): A dictionary with period as key and a dictionary as value.
            The dictionary contains the JSD and top shifted items.

    Returns:
        None
    """
    periods = list(js_results.keys())
    jsd_scores = [js_results[d]['JSD'] for d in periods]

    plt.figure(figsize=(15, 5))
    plt.plot(periods, jsd_scores, marker='o')
    plt.title("Jensen-Shannon Divergence Between Periods")
    plt.xlabel("Periods")
    plt.ylabel("JSD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot top-N shifting items
def plot_items_jsd_by_period(js_results, top_n=10, cols=3):
    """
    Plot the top-N shifting items between two periods.

    Parameters:
        js_results (dict): A dictionary with period as key and a dictionary as value.
            The dictionary contains the JSD and top shifted items.
        top_n (int): The number of top shifted items to plot.
        cols (int): The number of columns in the plot.

    Returns:
        None
    """
    num_periods = len(js_results)
    rows = math.ceil(num_periods / cols)

    # Find global max contribution across all periods
    global_max = max(
        max((item['contribution'] for item in result['top_shifted_items'][:top_n]), default=0)
        for result in js_results.values()
    )

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for idx, (decade, result) in enumerate(js_results.items()):
        ax = axes[idx]
        top_words = result['top_shifted_items'][:top_n]
        labels = [
            item['item'].replace("in_", "").replace("de_", "").replace("bo_", "").replace("lo_", "")
            for item in top_words
        ]
        values = [item['contribution'] for item in top_words]

        colors = [
            "lightgreen" if item['item'].startswith("in_") else 
            "lightcoral" if item['item'].startswith("de_") else 
            "darkgreen" if item['item'].startswith("bo_") else
            "darkred" if item['item'].startswith("lo_") else
            "purple"
            for item in top_words
        ]

        ax.barh(labels, values, color=colors)
        ax.invert_yaxis()

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
#----------------------------------------------------------------------------------

# Compute JSD of slot fillers across periods
def sfillers_jsd_by_period(df, 
                           word_col='chi_amod',
                           period_col='subfolder', 
                           min_count=0, 
                           top_n=10,
                           weight_df_path=None):
    """
    Compute the Jensen-Shannon Divergence (JSD) of slot fillers
    across periods.

    Parameters:
        df (pd.DataFrame): A DataFrame with period_col and word_col.
        word_col (str): Name of the column containing the syntactic fillers.
        period_col (str): Name of the column containing the period information.
        min_count (int): Minimum combined count across both periods to keep a slot.
        top_n (int): Number of top shifted items to return for each period pair.
        weight_df_path (str, optional): Path to a CSV file containing weights for each slot and period.

    Returns:
        dict: {period2: {'JSD': float, 'top_shifted_items': list[dict]}}
    """
    output = {}
    periods = sorted(df[period_col].dropna().unique())

    for period in range(1, len(periods)):
        period_1, period_2 = periods[period - 1], periods[period]
        vocab_1 = df[df[period_col] == period_1][word_col].value_counts()
        vocab_2 = df[df[period_col] == period_2][word_col].value_counts()

        vocab = sorted(set(vocab_1.index) | set(vocab_2.index))
        vocab = [w for w in vocab if vocab_1.get(w, 0) + vocab_2.get(w, 0) >= min_count]

        distribution_1 = np.array([vocab_1.get(w, 0) for w in vocab], dtype=float)
        distribution_2 = np.array([vocab_2.get(w, 0) for w in vocab], dtype=float)

        if distribution_1.sum() == 0 or distribution_2.sum() == 0:
            continue

        distribution_1 /= distribution_1.sum()
        distribution_2 /= distribution_2.sum()

        # Compute JSD
        jsd = cal_jsd(distribution_1, distribution_2)

        # Decompose the jsd to individual items
        contrib = cal_contrib_jsd(distribution_1, distribution_2, vocab)

        output[period_2] = {
            'JSD': jsd,
            'top_shifted_items': [item for item in contrib if item['contribution'] > 0][:top_n]
        }

    # Apply weight after the raw output has been created
    if weight_df_path is not None:
        weight_df = pd.read_csv(weight_df_path, index_col=0)

        # Normalize index and columns
        weight_df.index = weight_df.index.astype(str)
        weight_df.columns = weight_df.columns.astype(str)

        if word_col not in weight_df.index:
            raise ValueError(
                f"word_col='{word_col}' not found in weight_df index. "
                f"Available rows include: {list(weight_df.index[:10])}"
            )

        for year, values in output.items():
            year_str = str(year)

            if year_str not in weight_df.columns:
                raise ValueError(
                    f"Year '{year_str}' not found in weight_df columns."
                )

            weight = float(weight_df.loc[word_col, year_str])

            # Weight final JSD
            values["JSD"] = values["JSD"] * weight

            # Weight individual top filler contributions
            for item in values["top_shifted_items"]:
                item["contribution"] = item["contribution"] * weight
    return output

#----------------------------------------------------------------------------------
def _sort_periods(periods):
    """
    Sort periods numerically when possible.
    """
    try:
        return sorted(periods, key=lambda x: int(x))
    except Exception:
        return sorted(periods)


def _format_period_label(p1, p2):
    """
    Format period pair safely.
    """
    try:
        return f"{int(p2)}"
    except Exception:
        return f"{p2}"


def _parse_filler_cell(x):
    """
    Convert one dataframe cell into a list of fillers.

    Handles:
    - "['aboard/A', 'good/A']" -> ['aboard/A', 'good/A']
    - ['aboard/A', 'good/A'] -> ['aboard/A', 'good/A']
    - NaN -> []
    - single string -> [string]
    """
    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    if isinstance(x, str):
        x = x.strip()

        if x in ["", "[]", "nan", "None"]:
            return []

        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
            else:
                return [parsed]
        except Exception:
            return [x]

    return [x]

# Compute the consecutive JSD of the slots
def consecutive_jsd(
    temp_slot_df,
    period_col="subfolder",
    slot_col=None,
    mode="all",
    all_periods=None
):
    """
    Compute consecutive Jensen-Shannon Divergence for one slot.

    Modes
    -----
    mode="all":
        Compute JSD only between adjacent periods in the full period sequence.
        If either period has no retained data, skip that pair.

        Example:
            data in 1880 and 1900, but not 1890
            -> skip 1880-1890
            -> skip 1890-1900
            -> no 1880-1900 comparison

    mode="data_only":
        Compute JSD between adjacent available periods for that slot.
        Missing periods are ignored.

        Example:
            data in 1880 and 1900, but not 1890
            -> compute 1880-1900

    Parameters
    ----------
    temp_slot_df : pd.DataFrame
        Exploded dataframe with one row per slot filler occurrence.

    period_col : str
        Column containing periods.

    slot_col : str
        Column containing slot fillers.

    mode : {"all", "data_only"}
        JSD computation mode.

    all_periods : list, optional
        Full chronological period sequence. Required for true mode="all".

    Returns
    -------
    pd.DataFrame
        Columns: Period1, Period2, JSD
    """
    mode = mode.lower()
    assert mode in ["all", "data_only"], (
        f"mode must be either 'all' or 'data_only', but got {mode}"
    )

    if slot_col is None:
        raise ValueError("slot_col must be provided.")

    # Keep only valid period + filler rows
    work = temp_slot_df[[period_col, slot_col]].dropna(subset=[period_col, slot_col]).copy()

    # If no data survives filtering, return empty result
    if work.empty:
        return pd.DataFrame(columns=["Period1", "Period2", "JSD"])

    # Frequency table: period × filler
    freq = pd.crosstab(work[period_col], work[slot_col]).astype(float)

    if mode == "all":
        if all_periods is None:
            raise ValueError(
                "For mode='all', you must provide all_periods, "
                "e.g. list(range(1810, 2010, 10))."
            )

        all_periods = _sort_periods(all_periods)
        freq = freq.reindex(all_periods, fill_value=0)

    elif mode == "data_only":
        # Only keep periods where this slot has retained filler data
        row_sums = freq.sum(axis=1)
        freq = freq.loc[row_sums > 0]

    # If fewer than two periods remain, no JSD can be computed
    if len(freq.index) < 2:
        return pd.DataFrame(columns=["Period1", "Period2", "JSD"])

    row_sums = freq.sum(axis=1)

    results = []
    periods = freq.index.tolist()

    for i in range(1, len(periods)):
        period_1 = periods[i - 1]
        period_2 = periods[i]

        sum_1 = row_sums.loc[period_1]
        sum_2 = row_sums.loc[period_2]

        # Critical rule:
        # Do not compute JSD if either side has no data.
        if sum_1 == 0 or sum_2 == 0:
            continue

        distribution_1 = (freq.loc[period_1] / sum_1).to_numpy()
        distribution_2 = (freq.loc[period_2] / sum_2).to_numpy()

        jsd = cal_jsd(distribution_1, distribution_2)

        results.append({
            "Period1": period_1,
            "Period2": period_2,
            "JSD": jsd
        })

    return pd.DataFrame(results)

def compute_consecutive_JSD_dict(
    all_sfillers_csv_path,
    min_freq=1,
    mode="all",
    period_col="subfolder",
    exception_cols=("id", "subfolder", "target"),
    all_periods=None
):
    """
    Compute consecutive JSD for all slot-filler columns in a CSV file.

    Modes
    -----
    mode="all":
        Use the full chronological period sequence.
        Missing-data pairs are skipped.

    mode="data_only":
        Use only available periods for each slot.
        This may produce comparisons such as 1880-1900.

    Parameters
    ----------
    all_sfillers_csv_path : str
        Path to CSV file containing all slot fillers.

    min_freq : int
        Minimum total frequency of a filler across all periods.

    mode : {"all", "data_only"}
        JSD computation mode.

    period_col : str
        Period column name.

    exception_cols : tuple
        Non-slot columns to exclude.

    all_periods : list, optional
        Full period sequence. If None, inferred from the CSV.

    Returns
    -------
    dict
        {
            slot_type: {
                "period1-period2": JSD,
                ...
            },
            ...
        }
    """
    mode = mode.lower()
    assert mode in ["all", "data_only"], (
        f"mode must be either 'all' or 'data_only', but got {mode}"
    )

    all_sfillers_df = pd.read_csv(all_sfillers_csv_path, encoding="utf-8")

    # Full corpus period grid.
    # This is what makes mode="all" truly different from mode="data_only".
    if all_periods is None:
        all_periods = _sort_periods(all_sfillers_df[period_col].dropna().unique())
    else:
        all_periods = _sort_periods(all_periods)

    # Slot columns
    cols = [c for c in all_sfillers_df.columns if c not in exception_cols]

    output = {}

    for col in cols:
        # Keep period + one slot column
        df_temp = all_sfillers_df[[period_col, col]].copy()

        # Convert each cell to list
        df_temp[col] = df_temp[col].apply(_parse_filler_cell)

        # Explode list of fillers
        df_temp = (
            df_temp
            .explode(col, ignore_index=True)
            .dropna(subset=[period_col, col])
            .reset_index(drop=True)
        )

        # Remove empty string fillers, if any
        df_temp = df_temp[df_temp[col].astype(str).str.strip() != ""]

        # Apply global filler frequency threshold
        if not df_temp.empty:
            filler_freq = df_temp[col].value_counts()
            df_temp = df_temp[df_temp[col].map(filler_freq).fillna(0) >= min_freq]

        # Compute JSD
        consecutive_jsd_df = consecutive_jsd(
            temp_slot_df=df_temp,
            period_col=period_col,
            slot_col=col,
            mode=mode,
            all_periods=all_periods if mode == "all" else None
        )

        # Convert dataframe to nested dict
        jsd_dict = {
            _format_period_label(row.Period1, row.Period2): float(row.JSD)
            for row in consecutive_jsd_df.itertuples(index=False)
        }

        output[col] = jsd_dict

    return output

def compute_weighted_consecutive_JSD_dict(
    consecutive_JSD_dictionary: Dict[str, Dict[str, float]],
    support_dict: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Weight the consecutive JSD values by the support values for each slot and period pair.

    Parameters:
        consecutive_JSD_dictionary (Dict[str, Dict[str, float]]): Dictionary with slots as keys and values are dicts
                                                                with period pairs as keys and JSD values as values. 
        support_dict (Dict[str, Dict[str, float]]): Dictionary with slots as keys and values are dicts
                                                    with period pairs as keys and support values as values.

    Returns:
        Dict[str, Dict[str, float]]: Weighted dictionary with slots as keys and values are dicts
                                     with period pairs as keys and weighted JSD values as values.
    """

    weighted_dict = {}
    
    for slot, jsd_values in consecutive_JSD_dictionary.items():
        weighted_dict[slot] = {}
        
        for period_pair, jsd in jsd_values.items():
            support = support_dict.get(slot, {}).get(period_pair, 0)
            weighted_jsd = jsd * support
            weighted_dict[slot][period_pair] = weighted_jsd
    
    return weighted_dict

