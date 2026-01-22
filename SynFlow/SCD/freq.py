from collections import defaultdict, OrderedDict
import os
import json
import pandas as pd
import plotly.express as px
import random
import numpy as np



def count_keyword_tokens_by_period(corpus_path, keyword_string, 
                                   fname_pattern):
    # Use custom pattern to extract year from full filename
    counts_by_period = defaultdict(int)

    # Loop through each subfolder
    for subfolder in os.listdir(corpus_path):
        subfolder_path = os.path.join(corpus_path, subfolder)
        # Loop through each file in the subfolder
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            match = fname_pattern.search(filename) 

            if not match:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        counts_by_period[subfolder] += line.count(keyword_string)
            except Exception as e:
                print(f"Warning: failed to read {file_path}: {e}")

    return dict(sorted(counts_by_period.items()))

#-------------------------------------------------------------------------------------------------
# SLOT LEVEL
# Compute the frequency of ALL slots in each period
def freq_all_slots_by_period(json_path):
    # giữ nguyên thứ tự khóa như trong file
    with open(json_path, "r") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    keys = [k.strip() for k in data.keys()]          # làm sạch nhưng KHÔNG đổi thứ tự
    rows = [data[k] for k in keys]                   # lấy theo đúng order
    df = pd.DataFrame.from_records(rows, index=keys) # index = chuỗi năm theo file
    return df.fillna(0).astype(float)

# Compute the frequency (normalized by the number of tokens in that period) of ALL slots
def freq_all_slots_by_period_normalised_token_counts(json_path, normalized = False, 
                                                     token_counts = None):
    """
    Compute the frequency (normalized by the number of tokens in that period) of all slots

    Parameters:
        json_path (str): Path to JSON file (period → slot counts).
        normalized (bool): Normalize frequency by number of tokens.
        token_counts (dict): {period: token count} for normalization.

    Returns:
        df_long (pd.DataFrame): A long-form DataFrame with columns 'Period', 'Slot Type', and 'Frequency'.
    """
    df = freq_all_slots_by_period(json_path)

    # Normalised by the number of token counts in that period
    if normalized:
        if token_counts is None:
            raise ValueError("token_counts must be provided when normalized=True")
        token_counts_str = {str(k).replace("_", "-"): v for k, v in token_counts.items()}
        missing = set(df.index) - set(token_counts_str.keys())
        if missing:
            raise ValueError(f"Missing token counts for: {missing}")
        for period in df.index:
            df.loc[period] /= token_counts_str[period]

    df_long = df.reset_index().melt(id_vars="index", var_name="Slot Type", value_name="Frequency")
    df_long.rename(columns={"index": "Period"}, inplace=True)
    df_long["Period"] = pd.Categorical(df_long["Period"], categories=sorted(df.index), ordered=True)

    return df_long
# Compute the relative frequency of ALL slots in each period
def freq_all_slots_by_period_relative(json_path):
    """
    Compute the relative frequency of all slots.

    Parameters:
        json_path (str): Path to JSON file (period → slot counts).

    Returns:
        df_long (pd.DataFrame): A long-form DataFrame with columns 'Period', 'Slot Type', and 'Frequency'.
    """
    df = freq_all_slots_by_period(json_path)

    row_sum = df.sum(axis=1)
    df_rel = df.div(row_sum.replace(0, np.nan), axis=0).fillna(0.0)

    df_long = (df_rel.reset_index()
                     .melt(id_vars="index", var_name="Slot Type", value_name="Frequency")
                     .rename(columns={"index": "Period"}))
    df_long["Period"] = pd.Categorical(df_long["Period"],
                                       categories=sorted(df.index),
                                       ordered=True)
    return df_long

#  Plot the distribution of the union of TOP-N slots across periods
def freq_top_union_slots_by_period(json_path, top_n=10, relative=False,
                                    normalized=False, token_counts=None):
    """
    Compute the distribution of the union of top-N slots across periods.

    Parameters:
        json_path (str): Path to JSON with slot distributions per period.
        top_n (int): Number of top slot types per period to include.
        relative (bool): If True, compute relative frequency.
        normalized (bool): If True, normalize by token_counts.
        token_counts (dict): {period: token count} for normalization.

    Returns:
        df_long (pd.DataFrame): A long-form DataFrame with columns 'Period', 'Slot Type', and 'Frequency'.
    """

    if relative and normalized:
        raise ValueError("Choose either relative=True or normalized=True, not both.")
    
    # Get the frequency of all slots
    df = freq_all_slots_by_period(json_path)

    # Compute the union
    top_n_union = set()
    for period in df.index:
        top_slots = df.loc[period].sort_values(ascending=False).head(top_n).index
        top_n_union.update(top_slots)

    # Sort the slot types alphabetically for better plotting
    top_n_union = sorted(top_n_union, key=lambda s: str(s).lower())

    df_filtered = df[list(top_n_union)].astype(float)

    # Normalised by the number of token counts in that period
    if normalized:
        if token_counts is None:
            raise ValueError("token_counts must be provided when normalized=True")
        token_counts_str = {str(k).replace("_", "-"): v for k, v in token_counts.items()}

        # Handle missing token counts
        missing = set(df_filtered.index) - set(token_counts_str.keys())
        if missing:
            raise ValueError(f"Missing token counts for: {missing}")
        
        for period in df_filtered.index:
            df_filtered.loc[period] /= token_counts_str[period]

    # Relative frequency
    if relative:
        row_sum = df_filtered.sum(axis=1)
        df_filtered = df_filtered.div(row_sum.replace(0, np.nan), axis=0).fillna(0.0)

    df_long = df_filtered.reset_index().melt(id_vars="index", var_name="Slot Type", value_name="Frequency")
    df_long.rename(columns={"index": "Period"}, inplace=True)
    df_long["Period"] = pd.Categorical(df_long["Period"], categories=sorted(df.index), ordered=True)

    return df_long
def plot_freq_top_union_slots_by_period(json_path, top_n=10, relative=False,
                                        normalized=False, token_counts=None):
    
    """
    Plot the distribution of the union of top-N frequent slots across periods.

    Parameters:
        json_path (str): Path to JSON with slot distributions per period.
        top_n (int): Number of top slot types per period to include.
        normalized (bool): If True, normalize by token_counts.
        token_counts (dict): {period: token count} for normalization.

    Returns:
        fig (plotly.graph_objs.Figure): The plotted figure.
    """
    if relative and normalized:
        raise ValueError("Choose either relative=True or normalized=True, not both.")
    
    df_long = freq_top_union_slots_by_period(json_path, top_n, relative, normalized, token_counts)

    title = f"Frequencies of top-{top_n} Slots by Period (Union Set)"
    if normalized:
        title += " (Normalized)"

    if relative:
        title += " (Relative)"

    fig = px.line(
        df_long,
        x="Period",
        y="Frequency",
        color="Slot Type",
        markers=True,
        title=title,
        line_group="Slot Type"
    )

    fig.update_layout(
        xaxis_title="Period",
        yaxis_title=("Frequency per occurrence of target token " if normalized 
                     else "Relative Frequency" if relative 
                     else "Raw Count"),
        legend_title="Slot Type",
        height=500,
        width=1000
    )
    fig.show()

#-----------------------------------------------------------------------------------
# SLOT FILLER LEVEL
# Plot the distribution of the union of top-N slot fillers across periods
def freq_top_union_sfillers_by_period(csv_path, slot_type=None, top_n=10,
                                           normalized=False, time_col=None):

    """
    Compute the distribution of the union of top-N slot fillers across periods.

    Parameters:
        csv_path (str): Path to CSV file with slot fillers per period.
        slot_type (str): Name of the column containing the slot type.
        top_n (int): Number of top slot fillers per period to include in union.
        normalized (bool): Normalize frequency by number of documents.
        time_col (str): Name of the column containing the period information.

    Returns:
        df_long (pd.DataFrame): A long-form DataFrame with columns 'Period', 'Slot Type', and 'Frequency'.
    """
    if slot_type is None:
        raise ValueError("slot_type must be specified")
    if time_col is None:
        raise ValueError("time_col must be specified")

    # Load and prepare data ---
    df = pd.read_csv(csv_path)

    top_n_ovr = set()
    for period in df[time_col].dropna().unique():
        top_n_period = df[df[time_col] == period][slot_type].value_counts().nlargest(top_n).index
        top_n_ovr.update(top_n_period)

    df_top = df[df[slot_type].isin(top_n_ovr)]

    # Compute frequency ---
    if normalized:
        token_counts = df_top.groupby(time_col)['id'].nunique().reset_index(name='token_count')
        count_df = df_top.groupby([time_col, slot_type]).size().reset_index(name='count')
        count_df = count_df.merge(token_counts, on=time_col)
        count_df['norm_count'] = count_df['count'] / count_df['token_count']
    else:
        count_df = df_top.groupby([time_col, slot_type]).size().reset_index(name='count')
    
    return count_df

def plot_freq_top_union_sfillers_by_period(csv_path, slot_type=None, top_n=10,
                                           normalized=False, time_col=None):

    """
    Plot the distribution of the union of top-N slot fillers across periods.

    Parameters:
        csv_path (str): Path to CSV file with slot fillers per period.
        slot_type (str): Name of the column containing the slot type.
        top_n (int): Number of top slot fillers per period to include in union.
        normalized (bool): Normalize frequency by number of documents.
        time_col (str): Name of the column containing the period information.

    Returns:
        fig (plotly.graph_objs.Figure): The plotted figure.
    """
    count_df = freq_top_union_sfillers_by_period(csv_path, slot_type=slot_type, top_n=10,
                                           normalized=normalized, time_col=time_col)

    # Assign y_axis
    if normalized:
        y = 'norm_count'
        ylabel = 'Frequency per occurrence of target token'
    else: 
        y = 'count'
        ylabel = 'Absolute Frequency'

    # Assign numeric x-axis for correct time ordering
    count_df['time_num'] = count_df[time_col]
    tick_map = count_df.drop_duplicates('time_num')[['time_num', time_col]].sort_values('time_num')
    x_col = 'time_num'

    # Ensure consistent ordering of slot fillers when plotting
    filler_order = sorted(count_df[slot_type].unique(), key=lambda s: str(s).lower())
    count_df[slot_type] = pd.Categorical(count_df[slot_type], categories=filler_order, ordered=True)
    count_df = count_df.sort_values([slot_type, x_col])

    # Assign random colors to slot fillers
    unique_slot_fillers = sorted(count_df[slot_type].unique())
    random.seed(42)
    color_map = {slot_filler: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for slot_filler in unique_slot_fillers}

    # Plot using Plotly
    title = f"Frequencies of top-{top_n} {slot_type} per {time_col} (Union Set)"
    if normalized:
        title += " (Normalized)"

    fig = px.line(
        count_df,
        x=x_col,
        y=y,
        color=slot_type,
        color_discrete_map=color_map,
        title=title,
        markers=True,
        labels={
            x_col: time_col,
            y: ylabel,
        }
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=tick_map['time_num'],
            ticktext=tick_map[time_col]
        ),
        legend_title_text=slot_type,
        hovermode='x unified',
        height=500,
        width=1000
    )

    fig.show()