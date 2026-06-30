import ast
import math
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import jensenshannon

from SynFlow.Explorer import compute_saturating_support_from_sfiller_df
from SynFlow.const import DEFAULT_COLS

from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.stats.multitest import multipletests

#---------------------------------------------------------------
# Helper function to calculate JSD
def cal_jsd(distribution_a, distribution_b):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.

    Parameters:
        distribution_a (numpy array): First probability distribution.
        distribution_b (numpy array): Second probability distribution.

    Returns:
        float: The Jensen-Shannon divergence between p and q.
    """
    return jensenshannon(distribution_a, distribution_b, base=2)**2  # squared distance = divergence

# Helper function to decompose JSD
def cal_contrib_jsd(distribution_a, distribution_b, vocab):
    """
    Decompose a global JSD score into pointwise item contributions.

    Parameters:
        distribution_a (numpy array): First probability distribution.
        distribution_b (numpy array): Second probability distribution.
        vocab (list): List of slot types corresponding to the two distributions.

    Returns:
        list[dict]: Sorted list of dictionaries with keys ``item`` and ``contribution``.
    """
    distribution_mix = 0.5 * (distribution_a + distribution_b)
    pointwise_jsd = 0.5 * (distribution_a * np.log2(distribution_a / distribution_mix + 1e-12) + 
                           distribution_b * np.log2(distribution_b / distribution_mix + 1e-12))

    name_map = direction_prefix_map(
        vocab,
        distribution_a,
        distribution_b,
        prefix_increase="in_",
        prefix_decrease="de_",
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
def direction_prefix_map(vocab, distribution_a, distribution_b, prefix_increase="in_", prefix_decrease="de_",
                          prefix_born = 'bo_', prefix_lost = 'lo_', neutral=""):
    """
    Maps slot types to prefixed names based on the direction of the change.

    Parameters:
        vocab (list): List of slot types.
        distribution_a (numpy array): First probability distribution.
        distribution_b (numpy array): Second probability distribution.
        prefix_increase (str): Prefix for slot types that have increased in frequency.
        prefix_decrease (str): Prefix for slot types that have decreased in frequency.
        prefix_born (str): Prefix for slot types absent in distribution_a but present in distribution_b.
        prefix_lost (str): Prefix for slot types present in distribution_a but absent in distribution_b.
        neutral (str): Prefix for slot types that have not changed in frequency.

    Returns:
        dict: A dictionary with slot types as keys and prefixed names as values.
    """
    out = {}
    for i, slot in enumerate(vocab):
        if distribution_a[i] == 0 and distribution_b[i] > 0:
            out[slot] = f"{prefix_born}{slot}"
        elif distribution_a[i] > 0 and distribution_b[i] == 0:
            out[slot] = f"{prefix_lost}{slot}"
        elif distribution_a[i] == distribution_b[i]:
            out[slot] = f"{neutral}{slot}"
        elif distribution_a[i] > 0 and distribution_b[i] > 0:
            if distribution_b[i] > distribution_a[i]:
                out[slot] = f"{prefix_increase}{slot}"
            elif distribution_b[i] < distribution_a[i]:
                out[slot] = f"{prefix_decrease}{slot}"
    return out

#---------------------------------------------------------------
# Print JSD
def print_jsd_by_period(jsd_results):
    """
    Print the Jensen-Shannon Divergence and top shifted items for each period.

    Parameters:
        jsd_results (dict): A dictionary with period as key and a dictionary as value.
            The dictionary contains the JSD and top shifted items.

    Returns:
        None
    """
    for period, result in jsd_results.items():
        print(f"\n=== Shift to period {period} ===")
        print(f"Jensen-Shannon Divergence: {result['JSD']:.4f}")
        print("Top shifted items:")
        for item in result['top_shifted_items']:
            print(f"  {item['item']}: {item['contribution']:.4f}")

# Plot JSD
def plot_jsd_by_period(jsd_results):
    """
    Plot Jensen-Shannon Divergence values across period transitions.

    Parameters:
        jsd_results (dict): A dictionary with period as key and a dictionary as value.
            The dictionary contains the JSD and top shifted items.

    Returns:
        None
    """
    periods = list(jsd_results.keys())
    jsd_scores = [jsd_results[d]['JSD'] for d in periods]

    plt.figure(figsize=(15, 5))
    plt.plot(periods, jsd_scores, marker='o')
    plt.title("Jensen-Shannon Divergence Between Periods")
    plt.xlabel("Periods")
    plt.ylabel("JSD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot top-N shifting items
def plot_items_jsd_by_period(jsd_results, top_n=10, cols=3):
    """
    Plot the top-N shifting items between two periods.

    Parameters:
        jsd_results (dict): A dictionary with period as key and a dictionary as value.
            The dictionary contains the JSD and top shifted items.
        top_n (int): The number of top shifted items to plot.
        cols (int): The number of columns in the plot.

    Returns:
        None
    """
    num_periods = len(jsd_results)
    rows = math.ceil(num_periods / cols)

    # Find global max contribution across all periods
    global_max = max(
        max((item['contribution'] for item in result['top_shifted_items'][:top_n]), default=0)
        for result in jsd_results.values()
    )

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for idx, (decade, result) in enumerate(jsd_results.items()):
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

def sfillers_jsd_by_period(
    df,
    period_col="subfolder",
    slot_col="chi_amod",
    min_freq=1,
    mode="all",
    all_periods=None,
    top_n=10,
    weighting=False,
    k=20.0,
    include_zero_slots=False,
):
    """
    Compute filler-level JSD for one slot across consecutive periods.

    The input may contain either one filler per row or list-valued filler cells.
    List-valued cells are exploded internally before computing JSD. If
    ``weighting=True``, the raw JSD and item contributions are multiplied by
    saturating support computed from the same input column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``period_col`` and ``slot_col``.

    period_col : str
        Name of the column containing the period information.

    slot_col : str
        Name of the slot/filler column to compare.

    min_freq : int
        Minimum frequency of a filler within each period.
        If a filler occurs fewer than min_freq times in a period,
        it is treated as absent in that period.

    mode : {"all", "data_only"}
        Period-comparison mode. ``"all"`` compares adjacent periods in
        ``all_periods``; ``"data_only"`` uses adjacent periods with retained
        data.

    all_periods : list, optional
        Full period sequence. If None, inferred from ``df``.

    top_n : int
        Number of top shifted items to return for each period pair.

    weighting : bool
        If True, weight JSD scores by saturating support.

    k : float
        Support threshold for saturating weighting. Used only when
        ``weighting=True``.

    include_zero_slots : bool
        If True, include zero-support slots when computing support.

    Returns
    -------
    dict
        {
            period2: {
                "JSD": float,
                "top_shifted_items": list[dict]
            }
        }
    """

    if min_freq < 1:
        raise ValueError("`min_freq` must be >= 1.")

    mode = mode.lower()
    if mode not in {"all", "data_only"}:
        raise ValueError(
            f"`mode` must be either 'all' or 'data_only', but got {mode!r}."
        )

    if period_col not in df.columns:
        raise ValueError(f"Period column '{period_col}' not found in DataFrame.")

    if slot_col not in df.columns:
        raise ValueError(f"slot_col '{slot_col}' not found in DataFrame.")

    if all_periods is None:
        all_periods = _sort_periods(df[period_col].dropna().unique())
    else:
        all_periods = _sort_periods(all_periods)

    jsd_df = df[[period_col, slot_col]].copy()
    jsd_df[slot_col] = jsd_df[slot_col].apply(_parse_filler_cell)
    jsd_df = (
        jsd_df
        .explode(slot_col, ignore_index=True)
        .dropna(subset=[period_col, slot_col])
        .reset_index(drop=True)
    )
    jsd_df = jsd_df[jsd_df[slot_col].astype(str).str.strip() != ""]

    output = {}

    periods = all_periods

    for period in range(1, len(periods)):
        period_1, period_2 = periods[period - 1], periods[period]

        vocab_1 = jsd_df[jsd_df[period_col] == period_1][slot_col].value_counts()
        vocab_2 = jsd_df[jsd_df[period_col] == period_2][slot_col].value_counts()

        # Apply period-specific min_freq.
        # Fillers below min_freq are removed only from that period.
        vocab_1 = vocab_1[vocab_1 >= min_freq]
        vocab_2 = vocab_2[vocab_2 >= min_freq]

        # Vocabulary after period-specific filtering
        vocab = sorted(set(vocab_1.index) | set(vocab_2.index))

        distribution_a = np.array(
            [vocab_1.get(w, 0) for w in vocab],
            dtype=float
        )

        distribution_b = np.array(
            [vocab_2.get(w, 0) for w in vocab],
            dtype=float
        )

        if distribution_a.sum() == 0 or distribution_b.sum() == 0:
            continue

        distribution_a /= distribution_a.sum()
        distribution_b /= distribution_b.sum()

        # Compute JSD
        jsd = cal_jsd(distribution_a, distribution_b)

        # Decompose JSD into individual item contributions
        contrib = cal_contrib_jsd(distribution_a, distribution_b, vocab)

        output[period_2] = {
            "JSD": jsd,
            "top_shifted_items": [
                item for item in contrib
                if item["contribution"] > 0
            ][:top_n]
        }

    # Apply support weight after the raw output has been created.
    if weighting:
        support_df = df[[period_col, slot_col]].copy()
        support_df[slot_col] = support_df[slot_col].apply(_parse_filler_cell)

        saturating_support = compute_saturating_support_from_sfiller_df(
            sfiller_df=support_df,
            period_col=period_col,
            k=k,
            min_freq=min_freq,
            mode=mode,
            all_periods=all_periods,
            include_zero_slots=include_zero_slots,
        )

        for year, values in output.items():
            support_match = saturating_support[
                (saturating_support["slot"] == slot_col)
                & (saturating_support["period_2"].astype(str) == str(year))
            ]
            support_weight = (
                float(support_match["support_weight"].iloc[0])
                if not support_match.empty
                else 0.0
            )

            # Weight final JSD
            values["JSD"] = values["JSD"] * support_weight

            # Weight individual top filler contributions
            for item in values["top_shifted_items"]:
                item["contribution"] = item["contribution"] * support_weight

    return output

def plot_all_jsds_by_period(
    jsd_df: pd.DataFrame,
    slots: Optional[List[str]] = None,
    col_to_plot: str = None,
    layout: str = "combined",
    title: str = "Weighted JSD for all slots",
    y_label: str = "JSD",
    x_label: str = "Time Period",
    height: int = 700,
    width: int = 1100,
    save_path: Optional[str] = None,
):
    """
    Interactive time-series plot for slot-level JSD DataFrames.

    Parameters
    ----------
    jsd_df : pd.DataFrame
        DataFrame with at least ``slot``, ``period_1``, ``period_2``, and one
        numeric JSD value column. Common value columns are ``weighted_jsd`` and
        ``jsd``.

    slots : list, optional
        List of slot names to plot. If None, all slots are plotted.

    col_to_plot : str, optional
        Column to plot on the y-axis. If None, ``weighted_jsd`` is used when
        present, otherwise ``jsd``.

    layout : {"combined", "subplots", "dropdown"}
        - "combined": all slots on one interactive plot
        - "subplots": each slot in a separate subplot
        - "dropdown": one slot shown at a time, selected by dropdown

    title : str
        Figure title.

    y_label : str
        Y-axis label.

    x_label : str
        X-axis label.

    height : int
        Figure height.

    width : int
        Figure width.

    save_path : str, optional
        If provided, saves the figure as an interactive HTML file.
        Example: "slot_jsd_timeseries.html"

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    required_cols = {"slot", "period_1", "period_2"}
    missing_cols = required_cols - set(jsd_df.columns)
    if missing_cols:
        raise ValueError(
            f"jsd_df is missing required columns: {sorted(missing_cols)}"
        )

    if col_to_plot not in jsd_df.columns:
        raise ValueError(f"col_to_plot '{col_to_plot}' not found in jsd_df.")

    plot_df = jsd_df.copy()

    # Filter slots if specified
    if slots is not None:
        plot_df = plot_df[plot_df["slot"].isin(slots)]

    plot_df = plot_df.dropna(subset=["slot", "period_2", col_to_plot])

    if plot_df.empty:
        raise ValueError("No non-empty slot time series to plot.")

    slot_names = list(plot_df["slot"].drop_duplicates())
    layout = layout.lower()

    if layout == "combined":
        fig = go.Figure()

        for slot_name in slot_names:
            slot_df = plot_df[plot_df["slot"] == slot_name].copy()
            slot_df["_period_sort"] = slot_df["period_2"].map(_period_sort_value)
            slot_df = slot_df.sort_values("_period_sort")

            fig.add_trace(
                go.Scatter(
                    x=slot_df["period_2"],
                    y=slot_df[col_to_plot],
                    mode="lines+markers",
                    name=slot_name,
                    hovertemplate=(
                        f"<b>{slot_name}</b><br>"
                        "Period: %{x}<br>"
                        f"{y_label}: %{{y:.4f}}"
                        "<extra></extra>"
                    )
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=width,
            height=height,
            hovermode="closest",
            template="plotly_white",
            legend_title="Slot",
        )

    elif layout == "subplots":
        n_slots = len(slot_names)
        n_cols = 2
        n_rows = math.ceil(n_slots / n_cols)

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=slot_names
        )

        for idx, slot_name in enumerate(slot_names):
            row = idx // n_cols + 1
            subplot_col = idx % n_cols + 1

            slot_df = plot_df[plot_df["slot"] == slot_name].copy()
            slot_df["_period_sort"] = slot_df["period_2"].map(_period_sort_value)
            slot_df = slot_df.sort_values("_period_sort")

            fig.add_trace(
                go.Scatter(
                    x=slot_df["period_2"],
                    y=slot_df[col_to_plot],
                    mode="lines+markers",
                    name=slot_name,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{slot_name}</b><br>"
                        "Period: %{x}<br>"
                        f"{y_label}: %{{y:.4f}}"
                        "<extra></extra>"
                    )
                ),
                row=row,
                col=subplot_col
            )

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            hovermode="closest",
            template="plotly_white",
        )

        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)

    elif layout == "dropdown":
        fig = go.Figure()

        for idx, slot_name in enumerate(slot_names):
            slot_df = plot_df[plot_df["slot"] == slot_name].copy()
            slot_df["_period_sort"] = slot_df["period_2"].map(_period_sort_value)
            slot_df = slot_df.sort_values("_period_sort")

            fig.add_trace(
                go.Scatter(
                    x=slot_df["period_2"],
                    y=slot_df[col_to_plot],
                    mode="lines+markers",
                    name=slot_name,
                    visible=(idx == 0),
                    hovertemplate=(
                        f"<b>{slot_name}</b><br>"
                        "Period: %{x}<br>"
                        f"{y_label}: %{{y:.4f}}"
                        "<extra></extra>"
                    )
                )
            )

        buttons = []

        for idx, slot_name in enumerate(slot_names):
            visible = [False] * len(slot_names)
            visible[idx] = True

            buttons.append(
                dict(
                    label=slot_name,
                    method="update",
                    args=[
                        {"visible": visible},
                        {"title": f"{title}: {slot_name}"}
                    ]
                )
            )

        fig.update_layout(
            title=f"{title}: {slot_names[0]}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=width,
            height=height,
            hovermode="closest",
            template="plotly_white",
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    x=1.02,
                    y=1,
                    xanchor="left",
                    yanchor="top"
                )
            ]
        )

    else:
        raise ValueError(
            f"layout must be 'combined', 'subplots', or 'dropdown', got '{layout}'"
        )

    # Save as interactive HTML
    if save_path is not None:
        fig.write_html(save_path)
        print(f"Interactive figure saved to {save_path}")

    return fig

#----------------------------------------------------------------------------------
def _sort_periods(periods):
    """
    Sort period labels numerically when possible.

    Parameters
    ----------
    periods : iterable
        Period labels such as ``1880`` or ``"1880"``.

    Returns
    -------
    list
        Sorted period labels in their original value types.
    """
    try:
        return sorted(periods, key=lambda x: int(x))
    except Exception:
        return sorted(periods)

def _period_sort_value(period):
    """
    Convert a period label to a sorting value when possible.
    """
    try:
        return int(period)
    except Exception:
        return period

def _format_period_label(period_1, period_2):
    """
    Format a period transition label.

    The current public dictionary format uses the second period of the pair as
    the transition key.

    Parameters
    ----------
    period_1
        First period in the transition. Kept for call-site symmetry.
    period_2
        Second period in the transition.

    Returns
    -------
    str
        String label for ``period_2``.
    """
    try:
        return f"{int(period_2)}"
    except Exception:
        return f"{period_2}"

def _parse_filler_cell(cell):
    """
    Convert one DataFrame cell into a list of fillers.

    Handles:
    - "['aboard/A', 'good/A']" -> ['aboard/A', 'good/A']
    - ['aboard/A', 'good/A'] -> ['aboard/A', 'good/A']
    - NaN -> []
    - single string -> [string]

    Parameters
    ----------
    cell
        Cell value from a slot-filler column.

    Returns
    -------
    list
        Parsed filler values.
    """
    if isinstance(cell, list):
        return cell

    if pd.isna(cell):
        return []

    if isinstance(cell, str):
        cell = cell.strip()

        if cell in ["", "[]", "nan", "None"]:
            return []

        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, list):
                return parsed
            else:
                return [parsed]
        except Exception:
            return [cell]

    return [cell]

# Compute the consecutive JSD of the slots
def consecutive_jsd(
    temp_slot_df,
    slot_col=None,
    period_col="subfolder",
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

    slot_col : str
        Column containing slot fillers.

    period_col : str
        Column containing periods.

    mode : {"all", "data_only"}
        JSD computation mode.

    all_periods : list, optional
        Full chronological period sequence. Required for true mode="all".

    Returns
    -------
    pd.DataFrame
        Columns: slot, period_1, period_2, jsd
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
        return pd.DataFrame(columns=["slot", "period_1", "period_2", "jsd"])

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
        return pd.DataFrame(columns=["slot", "period_1", "period_2", "jsd"])

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

        distribution_a = (freq.loc[period_1] / sum_1).to_numpy()
        distribution_b = (freq.loc[period_2] / sum_2).to_numpy()

        jsd = cal_jsd(distribution_a, distribution_b)

        results.append({
            "slot": slot_col,
            "period_1": period_1,
            "period_2": period_2,
            "jsd": jsd
        })

    return pd.DataFrame(results, columns=["slot", "period_1", "period_2", "jsd"])

def compute_consecutive_jsd_df(
    sfiller_df: pd.DataFrame,
    period_col="subfolder",
    min_freq=1,
    mode="all",
    all_periods=None
):
    """
    Compute consecutive JSD for all slot-filler columns in a DataFrame.

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
    sfiller_df : pd.DataFrame
        DataFrame containing all slot fillers.

    period_col : str
        Period column name.

    min_freq : int
        Minimum frequency of a filler within each period.
        If a filler has frequency < min_freq in a given period,
        it is treated as absent in that period.

    mode : {"all", "data_only"}
        JSD computation mode.

    all_periods : list, optional
        Full period sequence. If None, inferred from the DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: slot, period_1, period_2, jsd.
    """
    mode = mode.lower()
    if mode not in {"all", "data_only"}:
        raise ValueError(
            f"`mode` must be either 'all' or 'data_only', but got {mode!r}."
        )

    sfiller_data = sfiller_df.copy()

    if all_periods is None:
        all_periods = _sort_periods(sfiller_data[period_col].dropna().unique())
    else:
        all_periods = _sort_periods(all_periods)

    # Slot columns
    slot_cols = [
        c for c in sfiller_data.columns
        if c not in DEFAULT_COLS and c != period_col
    ]
    output_frames = []

    for slot_col in slot_cols:
        # Keep period + one slot column
        slot_df = sfiller_data[[period_col, slot_col]].copy()

        # Convert each cell to list
        slot_df[slot_col] = slot_df[slot_col].apply(_parse_filler_cell)

        # Explode list of fillers
        slot_df = (
            slot_df
            .explode(slot_col, ignore_index=True)
            .dropna(subset=[period_col, slot_col])
            .reset_index(drop=True)
        )

        # Remove empty string fillers, if any
        slot_df = slot_df[slot_df[slot_col].astype(str).str.strip() != ""]

        # Apply period-specific filler frequency threshold
        # Example:
        # If min_freq = 2 and filler "b" occurs once in period A,
        # then "b" is removed from period A only.
        # If "b" occurs 5 times in period B, it is still kept in period B.
        if not slot_df.empty and min_freq > 1:
            period_filler_freq = (
                slot_df
                .groupby([period_col, slot_col])[slot_col]
                .transform("size")
            )

            slot_df = slot_df[period_filler_freq >= min_freq]

        # Compute JSD
        consecutive_jsd_table = consecutive_jsd(
            temp_slot_df=slot_df,
            slot_col=slot_col,
            period_col=period_col,
            mode=mode,
            all_periods=all_periods if mode == "all" else None
        )

        if not consecutive_jsd_table.empty:
            output_frames.append(consecutive_jsd_table)

    if not output_frames:
        return pd.DataFrame(columns=["slot", "period_1", "period_2", "jsd"])

    return pd.concat(output_frames, ignore_index=True)

def multiply_consecutive_jsd_saturating_support(
    consecutive_jsd: pd.DataFrame,
    saturating_support: pd.DataFrame,
) -> pd.DataFrame:
    """
    Multiply consecutive JSD values by matching support weights.

    Parameters
    ----------
    consecutive_jsd : pd.DataFrame
        DataFrame with columns ``slot``, ``period_1``, ``period_2``, and ``jsd``.

    saturating_support : pd.DataFrame
        DataFrame with columns ``slot``, ``period_1``, ``period_2``,
        ``support_count``, and ``support_weight``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``slot``, ``period_1``, ``period_2``, ``jsd``,
        ``support_count``, ``support_weight``, and ``weighted_jsd``.
    """
    output_cols = [
        "slot",
        "period_1",
        "period_2",
        "jsd",
        "support_count",
        "support_weight",
        "weighted_jsd",
    ]

    if consecutive_jsd.empty:
        return pd.DataFrame(columns=output_cols)

    support_cols = {
        "slot",
        "period_1",
        "period_2",
        "support_count",
        "support_weight",
    }
    missing_cols = support_cols - set(saturating_support.columns)
    if missing_cols:
        raise ValueError(
            f"saturating_support is missing required columns: {sorted(missing_cols)}"
        )

    weighted_consecutive_jsd = consecutive_jsd.merge(
        saturating_support[list(support_cols)],
        on=["slot", "period_1", "period_2"],
        how="left",
    )
    weighted_consecutive_jsd["support_count"] = (
        weighted_consecutive_jsd["support_count"].fillna(0.0)
    )
    weighted_consecutive_jsd["support_weight"] = (
        weighted_consecutive_jsd["support_weight"].fillna(0.0)
    )
    weighted_consecutive_jsd["weighted_jsd"] = (
        weighted_consecutive_jsd["jsd"]
        * weighted_consecutive_jsd["support_weight"]
    )

    return weighted_consecutive_jsd[output_cols]

def compute_weighted_consecutive_jsd_df(
    sfiller_df: pd.DataFrame,
    period_col: str = "subfolder",
    min_freq: int = 1,
    mode="all",
    all_periods=None,
    k: float = 20.0,
    include_zero_slots: bool = False,
) -> pd.DataFrame:
    """
    Compute support-weighted consecutive JSD for all slot columns.

    This function computes raw consecutive JSD with
    `compute_consecutive_jsd_df`, computes saturating support with
    `compute_saturating_support_from_sfiller_df``, and multiplies matching
    slot/period values.

    Parameters
    ----------
    sfiller_df : pd.DataFrame
        Slot-filler DataFrame. Metadata columns are excluded using
        ``DEFAULT_COLS``; all remaining columns are treated as slot columns.

    period_col : str
        Name of the period column.

    min_freq : int
        Minimum frequency of a filler within each period. Fillers below this
        threshold are treated as absent in that period.

    mode : {"all", "data_only"}
        Period-comparison mode used for both JSD and support.

    all_periods : list, optional
        Full period sequence. If None, inferred from ``sfiller_df``.

    k : float
        Support threshold. If support count is at least ``k``, the support
        weight is 1.0.

    include_zero_slots : bool
        If True, include slots with zero support in the support dictionary.

    Returns
    -------
    pd.DataFrame
        Columns: slot, period_1, period_2, jsd, support_count,
        support_weight, weighted_jsd.
    """

    consecutive_jsd = compute_consecutive_jsd_df(
        sfiller_df=sfiller_df,
        period_col=period_col,
        min_freq=min_freq,
        mode=mode,
        all_periods=all_periods,
    )

    saturating_support = compute_saturating_support_from_sfiller_df(
        sfiller_df=sfiller_df,
        period_col=period_col,
        min_freq=min_freq,
        mode=mode,
        all_periods=all_periods,
        k=k,
        include_zero_slots=include_zero_slots,
    )

    return multiply_consecutive_jsd_saturating_support(
        consecutive_jsd,
        saturating_support,
    )

#----------------------------------------------------------------------------------
# Permutation test
# Helper functions
def shuffle_period_labels(df_pair, period_col, rng):
    """
    Shuffle row-level period labels while preserving period sizes.

    The labels in ``period_col`` are permuted across rows in ``df_pair``.
    This preserves the number of rows assigned to each period, while breaking
    the observed association between period labels and slot fillers.

    Parameters
    ----------
    df_pair : pd.DataFrame
        Data for one period pair.

    period_col : str
        Column containing period labels.

    rng : numpy.random.Generator
        Random number generator used to shuffle labels.

    Returns
    -------
    pd.DataFrame
        Copy of ``df_pair`` with shuffled period labels.
    """
    out = df_pair.copy()

    labels = out[period_col].to_numpy(copy=True)
    rng.shuffle(labels)
    out[period_col] = labels

    return out

def chunk_list(x, chunk_size):
    """
    Yield consecutive chunks from a sequence.

    Parameters
    ----------
    x : sequence
        Input sequence to split.

    chunk_size : int
        Maximum number of items per chunk.

    Yields
    ------
    sequence
        Consecutive slices of ``x`` with length up to ``chunk_size``.
    """
    for i in range(0, len(x), chunk_size):
        yield x[i:i + chunk_size]

def jsd_stat_df_to_keyed_values(jsd_df, value_col):
    """
    Convert a JSD DataFrame to keyed values for permutation matching.

    Parameters
    ----------
    jsd_df : pd.DataFrame
        DataFrame with ``slot``, ``period_1``, ``period_2``, and ``value_col``.

    value_col : str
        JSD statistic column to use, usually ``jsd`` or ``weighted_jsd``.

    Returns
    -------
    dict
        Mapping ``(slot, period_1, period_2)`` to the selected JSD statistic.
    """
    required_cols = {"slot", "period_1", "period_2", value_col}
    missing_cols = required_cols - set(jsd_df.columns)
    if missing_cols:
        raise ValueError(
            f"jsd_df is missing required columns: {sorted(missing_cols)}"
        )

    return {
        (row["slot"], row["period_1"], row["period_2"]): float(row[value_col])
        for _, row in jsd_df.iterrows()
    }

def _permutation_consecutive_jsd_worker_chunk(
    df_pair,
    period_col,
    seeds,
    min_freq,
    k,
    weighting,
):
    """
    Run a chunk of consecutive-JSD permutations.

    This worker is submitted to a process pool. For each seed, it shuffles
    period labels within one period-pair DataFrame, recomputes the selected
    consecutive JSD statistic with ``mode="data_only"``, and converts the
    output to slot-transition-keyed numeric values.

    Parameters
    ----------
    df_pair : pd.DataFrame
        Data restricted to one adjacent period pair.

    period_col : str
        Column containing period labels.

    seeds : sequence of int
        Random seeds for the permutations handled by this worker.

    min_freq : int
        Minimum filler frequency within each period.

    k : float
        Support threshold for saturating weighting. Used only when
        ``weighting=True``.

    weighting : bool
        If True, use support-weighted JSD. If False, use raw JSD.

    Returns
    -------
    list[dict]
        One slot-transition-keyed JSD-statistic dictionary per permutation.
    """
    chunk_results = []
    value_col = "weighted_jsd" if weighting else "jsd"

    for seed in seeds:
        rng = np.random.default_rng(int(seed))

        shuffled_df_pair = shuffle_period_labels(
            df_pair=df_pair,
            period_col=period_col,
            rng=rng,
        )

        if weighting:
            null_df = compute_weighted_consecutive_jsd_df(
                sfiller_df=shuffled_df_pair,
                period_col=period_col,
                min_freq=min_freq,
                k=k,
                mode="data_only"
            )
        else:
            null_df = compute_consecutive_jsd_df(
                sfiller_df=shuffled_df_pair,
                period_col=period_col,
                min_freq=min_freq,
                mode="data_only"
            )

        null_values = jsd_stat_df_to_keyed_values(null_df, value_col)
        chunk_results.append(null_values)

    return chunk_results

def permutation_test_consecutive_jsd(
    sfiller_df,
    period_col="subfolder",
    all_periods=None,
    n_permutations=1000,
    min_freq=1,
    k=100,
    weighting=True,
    seed=42,
    keep_cols=None,
    n_jobs=8,
    chunk_size=50,
):
    """
    Run pairwise permutation tests for consecutive JSD.

    For each adjacent period pair in ``all_periods``, this function computes
    the observed consecutive JSD statistic for every slot column. If
    ``weighting=True``, the statistic is support-weighted JSD. If
    ``weighting=False``, the statistic is raw JSD. The null distribution is
    built by repeatedly shuffling period labels within each period pair and
    recomputing the same statistic. P-values are calculated as the proportion of
    null values greater than or equal to the observed value, with a standard
    plus-one correction. FDR correction is applied within each slot across
    adjacent period transitions.

    Parameters
    ----------
    sfiller_df : pd.DataFrame
        Slot-filler DataFrame. Metadata columns are excluded by
        ``compute_weighted_consecutive_jsd_df`` using ``DEFAULT_COLS``.

    period_col : str
        Column containing period labels.

    all_periods : list, optional
        Complete ordered period sequence. If None, periods are inferred from
        ``sfiller_df[period_col]`` and sorted with ``_sort_periods``.

    n_permutations : int
        Number of label-shuffle permutations per adjacent period pair.

    min_freq : int
        Minimum frequency of a filler within each period. Fillers below this
        threshold are treated as absent in that period.

    k : float
        Support threshold for saturating weighting. If support count is at
        least ``k``, the support weight is 1.0. Used only when
        ``weighting=True``.

    weighting : bool
        If True, run the permutation test on support-weighted JSD. If False,
        run it on raw JSD.

    seed : int
        Seed for the master random number generator that creates independent
        permutation seeds.

    keep_cols : list, optional
        Optional subset of columns to keep before running the test. The period
        column is always retained.

    n_jobs : int
        Number of worker processes used for permutations.

    chunk_size : int
        Number of permutation seeds submitted to each worker task.

    Returns
    -------
    pd.DataFrame
        One row per slot and adjacent period transition, with columns:
        ``slot``, ``period_1``, ``period_2``, ``statistic``,
        ``observed_statistic``, null-distribution summaries, ``p_value``,
        ``n_permutations``, ``q_value_fdr``, and ``significant_fdr_05``.
    """

    master_rng = np.random.default_rng(seed)
    value_col = "weighted_jsd" if weighting else "jsd"

    if keep_cols is not None:
        required_cols = {period_col}

        keep_cols = list(set(keep_cols) | required_cols)
        sfiller_df = sfiller_df[keep_cols].copy()

    if all_periods is None:
        all_periods = _sort_periods(sfiller_df[period_col].dropna().unique())
    else:
        all_periods = _sort_periods(all_periods)

    results = []

    for pair_id, (p1, p2) in enumerate(zip(all_periods[:-1], all_periods[1:])):

        print(f"Permutation testing for pair {pair_id + 1}/{len(all_periods) - 1}: {p1} -> {p2}")

        df_pair = sfiller_df[sfiller_df[period_col].isin([p1, p2])].copy()

        if df_pair[period_col].nunique() < 2:
            continue

        # 1. Observed statistic
        if weighting:
            obs_df = compute_weighted_consecutive_jsd_df(
                sfiller_df=df_pair,
                period_col=period_col,
                min_freq=min_freq,
                k=k,
                mode="data_only"
            )
        else:
            obs_df = compute_consecutive_jsd_df(
                sfiller_df=df_pair,
                period_col=period_col,
                min_freq=min_freq,
                mode="data_only"
            )

        obs_values = jsd_stat_df_to_keyed_values(obs_df, value_col)

        null_values = {
            slot_transition_key: []
            for slot_transition_key in obs_values.keys()
        }

        # 2. Generate independent seeds for permutations
        # The permutations run in parallel so instead sharing one random generator across workers (duplication), the main generator master_rng creates many independent seeds first, then each worker uses its assigned seeds to make reproducible shuffled datasets.
        seeds = master_rng.integers(
            low=0,
            high=2**32 - 1,
            size=n_permutations,
            dtype=np.uint32
        )

        seed_chunks = list(chunk_list(seeds, chunk_size))

        # 3. Run permutations in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _permutation_consecutive_jsd_worker_chunk,
                    df_pair,
                    period_col,
                    seed_chunk,
                    min_freq,
                    k,
                    weighting,
                )
                for seed_chunk in seed_chunks
            ]

            for future in as_completed(futures):
                chunk_results = future.result()

                for null_keyed_values in chunk_results:
                    for slot_transition_key in obs_values.keys():
                        null_values[slot_transition_key].append(
                            null_keyed_values.get(slot_transition_key, np.nan)
                        )

        # 4. Summarise null distribution
        for slot_transition_key, obs_value in obs_values.items():
            slot, period_1, period_2 = slot_transition_key

            arr = np.asarray(null_values[slot_transition_key], dtype=float)
            arr = arr[~np.isnan(arr)]

            if len(arr) == 0:
                p_value = np.nan
                null_mean = np.nan
                null_sd = np.nan
                null_q95 = np.nan
                null_q99 = np.nan
            else:
                p_value = (1 + np.sum(arr >= obs_value)) / (len(arr) + 1)
                null_mean = np.mean(arr)
                null_sd = np.std(arr, ddof=1)
                null_q95 = np.quantile(arr, 0.95)
                null_q99 = np.quantile(arr, 0.99)

            results.append({
                "slot": slot,
                "period_1": period_1,
                "period_2": period_2,
                "statistic": value_col,
                "weighting": weighting,
                "observed_statistic": obs_value,
                "null_mean": null_mean,
                "excess_over_null_mean": obs_value - null_mean,
                "null_sd": null_sd,
                "null_q95": null_q95,
                "null_q99": null_q99,
                "p_value": p_value,
                "n_permutations": len(arr),
            })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return pd.DataFrame(columns=[
            "slot",
            "period_1",
            "period_2",
            "statistic",
            "weighting",
            "observed_statistic",
            "null_mean",
            "excess_over_null_mean",
            "null_sd",
            "null_q95",
            "null_q99",
            "p_value",
            "n_permutations",
            "q_value_fdr",
            "significant_fdr_05",
        ])

    # FDR correction within each slot across adjacent transitions
    result_df["q_value_fdr"] = np.nan
    result_df["significant_fdr_05"] = False

    for slot, idx in result_df.groupby("slot").groups.items():
        pvals = result_df.loc[idx, "p_value"].to_numpy(dtype=float)

        valid_mask = ~np.isnan(pvals)

        if valid_mask.sum() == 0:
            continue

        # Benjamini–Yekutieli FDR correction
        reject, qvals, _, _ = multipletests(
            pvals[valid_mask],
            alpha=0.05,
            method="fdr_by"
        )

        valid_indices = result_df.loc[idx].index[valid_mask]

        result_df.loc[valid_indices, "q_value_fdr"] = qvals
        result_df.loc[valid_indices, "significant_fdr_05"] = reject

    return result_df
