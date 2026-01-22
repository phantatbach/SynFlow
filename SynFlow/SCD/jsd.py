import ast
from .freq import freq_all_slots_by_period_relative
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import seaborn as sns

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
        pd.Series: A Series with slot types as index and pointwise JSD as values,
            sorted in descending order.
    """
    distribution_mix = 0.5 * (distribution_1 + distribution_2)
    pointwise_jsd = 0.5 * (distribution_1 * np.log2(distribution_1 / distribution_mix + 1e-12) + 
                           distribution_2 * np.log2(distribution_2 / distribution_mix + 1e-12))
    contrib = pd.Series(pointwise_jsd, index=vocab).sort_values(ascending=False)

    # Build prefixed names based on direction (increase/decrease/no change)
    name_map = direction_prefix_map(vocab, distribution_1, distribution_2, prefix_in="in_", prefix_de="de_", prefix_born = 'bo_', prefix_lost = 'lo_', neutral="")
    contrib.index = [name_map[s] for s in contrib.index]

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
        for slot, score in result['top_shifted_items'].items():
            print(f"  {slot}: {score:.4f}")

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
            "lightgreen" if w.startswith("in_") else 
            "lightcoral" if w.startswith("de_") else 
            "darkgreen" if w.startswith("bo_") else
            "darkred" if w.startswith("lo_") else
            "purple"
            for w in top_words.index
        ]

        sns.barplot(
            x=top_words.values,
            y=top_words.index.str.replace("in_", "", regex=False).str.replace("de_", "", regex=False).str.replace("bo_", "", regex=False).str.replace("lo_", "", regex=False),
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
#----------------------------------------------------------------------------------
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
    for period in periods:
        top_slots = df.loc[period].sort_values(ascending=False).head(top_n).index
        slot_union.update(top_slots)
    slot_union = sorted(slot_union)

    output = {}
    for i in range(1, len(periods)):
        period_1, period_2 = periods[i - 1], periods[i]
        vocab_1 = df.loc[period_1][slot_union]
        vocab_2 = df.loc[period_2][slot_union]

        # Filter by min count
        vocab = [s for s in slot_union if vocab_1.get(s, 0) + vocab_2.get(s, 0) >= min_count]
        distribution_1 = np.array([vocab_1.get(s, 0) for s in vocab], dtype=float)
        distribution_2 = np.array([vocab_2.get(s, 0) for s in vocab], dtype=float)

        if distribution_1.sum() == 0 or distribution_2.sum() == 0:
            continue

        distribution_1 /= distribution_1.sum()
        distribution_2 /= distribution_2.sum()

        # Compute JSD
        jsd = cal_jsd(distribution_1, distribution_2)

        # Decompose the jsd to individual items
        contrib = cal_contrib_jsd(distribution_1, distribution_2, vocab)

        # Gán nhãn kết quả bằng period 2
        output[period_2] = {
            'JSD': jsd,
            'top_shifted_items': contrib[contrib > 0].head(10)
        }

    return output

# Compute JSD of slot fillers across periods
def sfillers_jsd_by_period(df, word_col='chi_amod', period_col='subfolder', min_count=0, top_n=10):
    """
    Compute the Jensen-Shannon Divergence (JSD) of slot fillers
    across periods.

    Parameters:
        df (pd.DataFrame): A DataFrame with period_col and word_col.
        word_col (str): Name of the column containing the syntactic fillers.
        period_col (str): Name of the column containing the period information.
        min_count (int): Minimum combined count across both periods to keep a slot.

    Returns:
        dict: {period2: {'JSD': float, 'top_shifted_items': pd.Series}}
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
            'top_shifted_items': contrib[contrib > 0].head(top_n)
        }

    return output

#----------------------------------------------------------------------------------
# Compute the consecutive JSD of the slots
def consecutive_jsd(temp_slot_df,
                    period_col="subfolder",
                    slot_col=str,
                    mode=str):
    """
    Compute the consecutive Jensen-Shannon Divergence (JSD) of a slot.

    Parameters:
        slot_df (pd.DataFrame): A DataFrame with columns 'subfolder' and 'slot'
        period_col (str): The column name containing the period information.
        slot_col (str): The column name containing the slot information.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Period1', 'Period2', and 'JSD'.
    """
    assert mode in ["all", "data_only"], f"mode must be either 'all' or 'data_only', but got {mode}" # mode: all, data_only

    freq = pd.crosstab(temp_slot_df[period_col], temp_slot_df[slot_col]).astype(float)  # frequency table: period × slot

    # Get all periods if mode == "all"
    if mode == "all":
        all_periods = sorted(temp_slot_df[period_col].unique())
        freq = freq.reindex(all_periods, fill_value=0)

    # normalize rows to probability distributions
    row_sums = freq.sum(axis=1)
    prob = freq.div(row_sums, axis=0)

    results = []
    periods = prob.index.tolist()
    for period in range(1, len(periods)):
        period_1, period_2 = periods[period-1], periods[period]
        # with data_only, remove the pair if one is 0
        if mode == "data_only" and (row_sums.loc[period_1] == 0 or row_sums.loc[period_2] == 0):
            continue
        # with all, remove the pair if both are 0
        if mode == "all" and (row_sums.loc[period_1] == 0 and row_sums.loc[period_2] == 0):
            continue
        
        distribution_1, distribution_2 = prob.iloc[period-1], prob.iloc[period]
        # union of slots already aligned by crosstab
        JSD = cal_jsd(distribution_2.values, distribution_1.values)
        results.append({
            "Period1": periods[period-1],
            "Period2": periods[period],
            "JSD": JSD
        })
    return pd.DataFrame(results)

# # Calculate consecutive pairs with positive frequencies
# def pairs_with_pos_freq(temp_slot_df):
#     """
#     Compute the number of pairs of consecutive slots with positive frequency.

#     Parameters:
#         temp_slot_df (pd.DataFrame): A DataFrame with columns 'subfolder', 'slot', and 'Frequency'.

#     Returns:
#         int: Number of pairs of consecutive slots with positive frequency.
#     """
#     freqs = temp_slot_df["Frequency"].to_numpy()
#     pair_count = ((freqs[:-1] > 0) & (freqs[1:] > 0)).sum()
#     return pair_count

# # Calculate the weighted total divergence of a slot
# def weighted_total_divergence_col(consecutive_jsd_df, freq_col_slot_by_period_df):
#     """
#     Compute the weighted total divergence for a given slot type column.

#     Parameters:
#         consecutive_jsd_df (pd.DataFrame): A DataFrame with columns 'Period1', 'Period2', and 'JSD'.
#         freq_col_slot_by_period_df (pd.DataFrame): A DataFrame with columns 'Period', 'Slot Type', and 'Frequency'.

#     Returns:
#         float: The weighted total divergence for the given slot type column.
#     """
#     total_divergence_col_val = 0
#     for row in consecutive_jsd_df.itertuples():
#         freq_period_1 = freq_col_slot_by_period_df.loc[freq_col_slot_by_period_df["Period"].astype(int) == row.Period1, "Frequency"].iloc[0]
#         freq_period_2 = freq_col_slot_by_period_df.loc[freq_col_slot_by_period_df["Period"].astype(int) == row.Period2, "Frequency"].iloc[0]

#         # Min = Stable
#         # Max = Big spike or drop
#         # Sum = Balance
#         weight_freq = min(freq_period_1, freq_period_2)

#         total_divergence_col_val += row.JSD * weight_freq

#     return total_divergence_col_val

# # Calculate the total divergence of all the slots
# def total_divergence_slots(slot_json_path, all_sfillers_csv_path, min_freq=1):
#     """
#     Compute the weighted total divergence for all slot types.

#     Parameters:
#         slot_json_path (str): Path to JSON file containing slot frequencies.
#         all_sfillers_csv_path (str): Path to CSV file containing all slot types.
#         min_freq (int): Minimum frequency threshold for a slot fillers

#     Returns:
#         pd.DataFrame: A DataFrame with columns 'Slot Type' and 'Weighted Total Divergence'.
#     """
#     all_sfillers_df = pd.read_csv(all_sfillers_csv_path, encoding="utf-8")  # Get the sfiller df of all the slots

#     # Calculate relative frequencies of all slots by period
#     rel_freq_all_slots_by_period_df = freq_all_slots_by_period_relative(json_path=slot_json_path)

#     # Get the list of slot columns
#     exception = ['id', 'subfolder', 'target']
#     cols = [c for c in all_sfillers_df.columns if c not in exception]

#     rows = []
#     for col in cols:
#         # Create a temporary df with only one col
#         df_temp = all_sfillers_df[['subfolder', col]].copy()
#         df_temp[col] = df_temp[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#         df_temp = (df_temp
#                    .explode(col, ignore_index=True)
#                    .dropna(subset=[col])
#                    .reset_index(drop=True))

#         # Keep only the slot fillers with total frequency >= min_freq
#         freq = df_temp[col].value_counts()
#         df_temp = df_temp[df_temp[col].map(freq) >= min_freq]

#         # Calculate JSD
#         consecutive_jsd_df = consecutive_jsd(df_temp, period_col='subfolder', slot_col=col)

#         # Calculate total divergence, weighted by frequencies of 2 periods
#         freq_col_slot_by_period_df = rel_freq_all_slots_by_period_df[rel_freq_all_slots_by_period_df["Slot Type"] == col].reset_index(drop=True)
#         pos_pairs_counts = pairs_with_pos_freq(freq_col_slot_by_period_df)

#         total_divergence = weighted_total_divergence_col(consecutive_jsd_df, freq_col_slot_by_period_df)

#         if pos_pairs_counts == 0:
#             normalised_total_divergence = 0
#         else:
#             normalised_total_divergence = total_divergence / pos_pairs_counts

#         rows.append((col, normalised_total_divergence))

#     total_divergence_df = (
#         pd.DataFrame(rows, columns=["Slot Type", "Weighted Total Divergence"])
#         .sort_values("Weighted Total Divergence", ascending=False, ignore_index=True)
#     )
#     return total_divergence_df

def consecutive_JSD_dict(all_sfillers_csv_path, 
                         min_freq=1,
                         mode='all'):
    """
    Compute the consecutive Jensen-Shannon Divergence (JSD) of all slot fillers
    in a given CSV file.

    Parameters:
        all_sfillers_csv_path (str): Path to the CSV file containing all slot fillers.
        min_freq (int): Minimum frequency of a slot filler to be kept.

    Returns:
        dict: A dictionary where each key is a slot type and each value is a dictionary
            containing the JSD values between consecutive periods for that slot type.
    """
    all_sfillers_df = pd.read_csv(all_sfillers_csv_path, encoding="utf-8")  # Get the sfiller df of all the slots

    # Get the list of slot columns
    exception = ['id', 'subfolder', 'target']
    cols = [c for c in all_sfillers_df.columns if c not in exception]
    
    # Create an empty dict
    consecutive_JSD_dict = {}

    for col in cols:
        # Create a temporary df with only one col
        df_temp = all_sfillers_df[['subfolder', col]].copy()
        df_temp[col] = df_temp[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df_temp = (df_temp
                   .explode(col, ignore_index=True)
                #    .dropna(subset=[col])
                   .reset_index(drop=True))

        # Keep only the slot fillers with total frequency >= min_freq
        freq = df_temp[col].value_counts()
        df_temp = df_temp[df_temp[col].map(freq) >= min_freq]

        # Calculate JSD
        consecutive_jsd_df = consecutive_jsd(df_temp, period_col='subfolder', slot_col=col, mode=mode)
        
        # Convert df to dict
        jsd_dict = {f"{int(row.Period1)}-{int(row.Period2)}": float(row.JSD)
                    for row in consecutive_jsd_df.itertuples(index=False)}
        
        consecutive_JSD_dict[col] = jsd_dict

    return consecutive_JSD_dict

# def summarise_jsd_dict(jsd_dict):
#     """
#     Input:
#         jsd_dict = {
#             slot_type: { "period1-period2": JSD, ... },
#             ...
#         }

#     Output:
#         DataFrame với cột: Sum, Positive Pair, Mean
#     """
#     rows = []
#     for slot, pairs in jsd_dict.items():
#         if pairs:  # dict con không rỗng
#             vals = list(pairs.values())
#             sum_JSD = sum(vals)
#             max_JSD = max(vals)
#             pos_pairs = len(vals)
#             mean_JSD = sum_JSD / pos_pairs if pos_pairs > 0 else 0
#         else:  # dict con rỗng
#             sum_JSD, pos_pairs, mean_JSD, max_JSD = 0, 0, 0, 0
#         rows.append((slot, pos_pairs, sum_JSD, mean_JSD, max_JSD))

#     df = pd.DataFrame(rows, columns=["Slot", "Positive Pairs", "Sum", "Mean", "Max"]).sort_values("Mean", ascending=False).reset_index(drop=True)

#     return df

