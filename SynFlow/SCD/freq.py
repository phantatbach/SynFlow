from collections import defaultdict
import os
import json
import pandas as pd
import plotly.express as px
import random
def count_keyword_tokens_by_period(corpus_path, keyword_string, pattern, mode='half_decade'):
    """
    Count how many times the keyword string appears in each time period.

    Parameters:
        corpus_path (str): Folder with text files.
        keyword_string (str): Target string (e.g., 'air\\tair\\tNOUN').
        mode (str): 'decade' or 'half_decade'.

    Returns:
        dict: {period (int): token count}
    """
    assert mode in {'decade', 'half_decade'}, "Mode must be 'decade' or 'half_decade'"

    # Use custom pattern to extract year from full filename
    pattern = pattern
    counts_by_period = defaultdict(int)

    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        match = pattern.search(filename) 

        if not match:
            continue

        year = int(match.group("year"))
        period = (year // 10) * 10 if mode == 'decade' else (year // 5) * 5

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    counts_by_period[period] += line.count(keyword_string)
        except Exception as e:
            print(f"Warning: failed to read {file_path}: {e}")

    return dict(sorted(counts_by_period.items()))

# Plot the distribution of the union of top-N slots across periods
def plot_top_n_union_slots(json_path, top_n=10, normalized=False, token_counts=None, chart_type="line"):
    """
    Plot the distribution of the union of top-n slots across periods as bar or line chart.

    Parameters:
        json_path (str): Path to JSON with slot distributions per period.
        top_n (int): Number of top slot types per period to include.
        normalized (bool): If True, normalize by token_counts.
        token_counts (dict): {period: token count} for normalization.
        chart_type (str): "line" or "bar"
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)

    top_n_union = set()
    for period in df.index:
        top_slots = df.loc[period].sort_values(ascending=False).head(top_n).index
        top_n_union.update(top_slots)

    df_filtered = df[list(top_n_union)].astype(float)

    # Normalised by the number of token counts in that period
    if normalized:
        if token_counts is None:
            raise ValueError("token_counts must be provided when normalized=True")
        token_counts_str = {str(k).replace("_", "-"): v for k, v in token_counts.items()}
        missing = set(df_filtered.index) - set(token_counts_str.keys())
        if missing:
            raise ValueError(f"Missing token counts for: {missing}")
        for period in df_filtered.index:
            df_filtered.loc[period] /= token_counts_str[period]

    df_long = df_filtered.reset_index().melt(id_vars="index", var_name="Slot Type", value_name="Frequency")
    df_long.rename(columns={"index": "Period"}, inplace=True)
    df_long["Period"] = pd.Categorical(df_long["Period"], categories=sorted(df.index), ordered=True)

    title = f"Top-{top_n} Slots Over Time"
    if normalized:
        title += " (Normalized)"

    if chart_type == "line":
        fig = px.line(
            df_long,
            x="Period",
            y="Frequency",
            color="Slot Type",
            markers=True,
            title=title,
            line_group="Slot Type"
        )
    elif chart_type == "bar":
        fig = px.bar(
            df_long,
            x="Period",
            y="Frequency",
            color="Slot Type",
            barmode="group",
            title=title
        )
    else:
        raise ValueError("chart_type must be 'line' or 'bar'")

    fig.update_layout(
        xaxis_title="Period",
        yaxis_title="Frequency per occurrence of target token " if normalized else "Raw Count",
        legend_title="Slot Type",
        height=500,
        width=1000
    )
    fig.show()

# Plot the distribution of the union of top-N slot fillers across periods
def plot_top_slot_fillers_by_period(csv_path, slot_type=None, top_n=10, normalized=False, time_col=None, chart_type='bar'):
    """
    Interactive chart (bar or line) of top slot fillers per period.

    Args:
        csv_path (str): CSV with columns 'id', slot_type, and time columns.
        slot_type (str): The specific type of slot
        top_n (int): Number of top adjectives per period to include in union.
        normalized (bool): Normalize frequency by number of documents.
        time_col (str): whatever the column name of the time column is, e.g. 'decade' or 'half_decade'.
        chart_type (str): 'bar' or 'line' chart.
    """
    assert chart_type in ['bar', 'line'], "chart_type must be 'bar' or 'line'"
    assert slot_type is not None, "slot_type must be specified"
    assert time_col is not None, "time_col must be specified"

    # --- Step 1: Load and prepare data ---
    df = pd.read_csv(csv_path)

    top_n_ovr = set()
    for period in df[time_col].dropna().unique():
        top_n_period = df[df[time_col] == period][slot_type].value_counts().nlargest(top_n).index
        top_n_ovr.update(top_n_period)

    df_top = df[df[slot_type].isin(top_n_ovr)]

    # --- Step 2: Compute frequency ---
    if normalized:
        token_counts = df_top.groupby(time_col)['id'].nunique().reset_index(name='token_count')
        count_df = df_top.groupby([time_col, slot_type]).size().reset_index(name='count')
        count_df = count_df.merge(token_counts, on=time_col)
        count_df['norm_count'] = count_df['count'] / count_df['token_count']
        y = 'norm_count'
        ylabel = 'Frequency per occurrence of target token'
    else:
        count_df = df_top.groupby([time_col, slot_type]).size().reset_index(name='count')
        y = 'count'
        ylabel = 'Absolute Frequency'

    # --- Step 3: Assign numeric x-axis for correct time ordering ---
    count_df['time_num'] = count_df[time_col]
    tick_map = count_df.drop_duplicates('time_num')[['time_num', time_col]].sort_values('time_num')
    x_col = 'time_num'

    # --- Step 4: Assign random colors to slot fillers ---
    unique_slot_fillers = sorted(count_df[slot_type].unique())
    random.seed(42)
    color_map = {slot_filler: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for slot_filler in unique_slot_fillers}

    # --- Step 5: Plot using Plotly ---
    title = f"Top {top_n} {slot_type} per {time_col} (union set)"
    if normalized:
        title += " (normalized)"

    if chart_type == 'bar':
        fig = px.bar(
            count_df,
            x=x_col,
            y=y,
            color=slot_type,
            color_discrete_map=color_map,
            title=title,
            labels={
                x_col: time_col,
                y: ylabel,
            }
        )
        fig.update_layout(barmode="group")

    elif chart_type == 'line':
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