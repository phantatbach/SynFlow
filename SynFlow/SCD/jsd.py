import math
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import jensenshannon
from statsmodels.stats.multitest import multipletests

from SynFlow.const import DEFAULT_COLS
from SynFlow.Explorer import compute_saturating_support_from_sfiller_df
from SynFlow.Explorer.sfiller_df import parse_filler_cell

#-------------------------------------------------------------------------------
# Core JSD Helpers
#-------------------------------------------------------------------------------
def cal_jsd(distribution_a: np.ndarray, distribution_b: np.ndarray) -> float:
    """
    Compute the Jensen-Shannon divergence between two probability distributions.

    Parameters:
        distribution_a (numpy array): First probability distribution.
        distribution_b (numpy array): Second probability distribution.

    Returns:
        float: The Jensen-Shannon divergence between p and q.
    """
    return jensenshannon(distribution_a, distribution_b, base=2)**2

def direction_prefix_map(
    vocab: Sequence[Any],
    distribution_a: np.ndarray,
    distribution_b: np.ndarray,
    prefix_increase: str = "in_",
    prefix_decrease: str = "de_",
    prefix_born: str = "bo_",
    prefix_lost: str = "lo_",
    neutral: str = "",
) -> Dict[Any, str]:
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

def cal_contrib_jsd(
    distribution_a: np.ndarray,
    distribution_b: np.ndarray,
    vocab: Sequence[Any],
) -> List[Dict[str, Any]]:
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

#-------------------------------------------------------------------------------
# Period And Validation Helpers
#-------------------------------------------------------------------------------
def _sort_periods(periods: Iterable[Any]) -> List[Any]:
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

def _period_sort_value(period: Any) -> Any:
    """
    Convert a period label to a sorting value when possible.
    """
    try:
        return int(period)
    except Exception:
        return period

def _format_period_label(period_1: Any, period_2: Any) -> str:
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

def _normalize_period_sequence(
    periods: Iterable[Any],
    sort: bool = True,
) -> List[str]:
    """Convert period labels to strings, optionally sorting them."""
    normalized_periods = [
        str(period)
        for period in periods
        if not pd.isna(period)
    ]
    if sort:
        return _sort_periods(normalized_periods)
    return normalized_periods

def _normalize_period_column(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """Return a copy with non-missing period labels converted to strings."""
    out = df.copy()
    period_mask = out[period_col].notna()
    out.loc[period_mask, period_col] = out.loc[period_mask, period_col].map(str)
    return out

def _validate_period_column(df: pd.DataFrame, period_col: str) -> None:
    """Validate that a DataFrame contains the requested period column."""
    if period_col not in df.columns:
        raise ValueError(f"Period column '{period_col}' not found in DataFrame.")

def _validate_min_freq(min_freq: int) -> None:
    """Validate a pair-specific minimum filler frequency."""
    if min_freq < 1:
        raise ValueError("min_freq must be >= 1.")

def _validate_positive_k(k: float) -> None:
    """Validate a positive support threshold."""
    if k <= 0:
        raise ValueError("k must be > 0.")

def _validate_jsd_mode(mode: str) -> str:
    """Validate and normalize a JSD period-comparison mode."""
    if not isinstance(mode, str):
        raise ValueError(
            f"`mode` must be either 'all' or 'data_only', but got {mode!r}."
        )

    normalized_mode = mode.lower()
    if normalized_mode not in {"all", "data_only"}:
        raise ValueError(
            f"`mode` must be either 'all' or 'data_only', but got {mode!r}."
        )
    return normalized_mode

def _validate_permutation_arguments(
    n_permutations: int,
    chunk_size: int,
    n_jobs: int,
) -> None:
    """Validate permutation execution arguments."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1.")

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

#-------------------------------------------------------------------------------
# Consecutive JSD
#-------------------------------------------------------------------------------
def consecutive_jsd(
    temp_slot_df: pd.DataFrame,
    slot_col: Optional[str] = None,
    period_col: str = "subfolder",
    min_freq: int = 1,
    mode: str = "all",
    all_periods: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """
    Compute consecutive Jensen-Shannon Divergence for one slot.

    Modes
    -----
    mode="all":
        Compute JSD only between adjacent periods in the full period sequence.
        If either period has no data after pair-level filtering, skip that pair.

        Example:
            data in 1880 and 1900, but not 1890
            -> skip 1880-1890
            -> skip 1890-1900
            -> no 1880-1900 comparison

    mode="data_only":
        Compute JSD between adjacent periods with raw filler data for that slot.
        Each pair is then filtered by mixed pair frequency; pairs without data
        on both sides after filtering are skipped.

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

    min_freq : int
        Minimum frequency of a filler across the mixed distribution of each
        compared period pair.

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
    if mode not in {"all", "data_only"}:
        raise ValueError(
            f"mode must be 'all' or 'data_only', got {mode!r}."
        )

    if slot_col is None:
        raise ValueError("slot_col must be provided.")

    _validate_min_freq(min_freq)

    # Keep only valid period + filler rows
    work = temp_slot_df[[period_col, slot_col]].dropna(subset=[period_col, slot_col]).copy()
    work = _normalize_period_column(work, period_col)

    # If no data survives filtering, return empty result
    if work.empty:
        return pd.DataFrame(columns=["slot", "period_1", "period_2", "jsd"])

    if mode == "all":
        if all_periods is None:
            raise ValueError(
                "For mode='all', you must provide all_periods, "
                "e.g. list(range(1810, 2010, 10))."
            )

        all_periods = _normalize_period_sequence(all_periods, sort=False)
        periods = list(all_periods)

    elif mode == "data_only":
        periods = _normalize_period_sequence(work[period_col].dropna().unique())

    # If fewer than two periods remain, no JSD can be computed
    if len(periods) < 2:
        return pd.DataFrame(columns=["slot", "period_1", "period_2", "jsd"])

    results = []

    for i in range(1, len(periods)):
        period_1 = periods[i - 1]
        period_2 = periods[i]

        pair_work = work[work[period_col].isin([period_1, period_2])].copy()
        if pair_work.empty:
            continue

        if min_freq > 1:
            mixed_filler_freq = pair_work.groupby(slot_col)[slot_col].transform("size")
            pair_work = pair_work[mixed_filler_freq >= min_freq]

        if pair_work.empty:
            continue

        freq = pd.crosstab(pair_work[period_col], pair_work[slot_col]).astype(float)
        freq = freq.reindex([period_1, period_2], fill_value=0.0)
        row_sums = freq.sum(axis=1)

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
    period_col: str = "subfolder",
    min_freq: int = 1,
    mode: str = "all",
    all_periods: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """
    Compute consecutive JSD for all slot-filler columns in a DataFrame.

    Modes
    -----
    mode="all":
        Use the full chronological period sequence.
        Missing-data pairs are skipped.

    mode="data_only":
        Use only periods with raw filler data for each slot. Pair-level filtering
        may still skip comparisons after mixed frequency filtering.
        This may produce comparisons such as 1880-1900.

    Parameters
    ----------
    sfiller_df : pd.DataFrame
        DataFrame containing all slot fillers.

    period_col : str
        Period column name.

    min_freq : int
        Minimum frequency of a filler across the mixed distribution of each
        compared period pair. Fillers below this threshold are treated as
        absent from that pair.

    mode : {"all", "data_only"}
        JSD computation mode.

    all_periods : list, optional
        Full period sequence. If None, inferred from the DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: slot, period_1, period_2, jsd.
    """
    _validate_period_column(sfiller_df, period_col)
    _validate_min_freq(min_freq)

    mode = _validate_jsd_mode(mode)

    sfiller_data = _normalize_period_column(sfiller_df, period_col)

    if all_periods is None:
        all_periods = _normalize_period_sequence(sfiller_data[period_col].dropna().unique())
    else:
        all_periods = _normalize_period_sequence(all_periods, sort=False)

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
        slot_df[slot_col] = slot_df[slot_col].apply(parse_filler_cell)

        # Explode list of fillers
        slot_df = (
            slot_df
            .explode(slot_col, ignore_index=True)
            .dropna(subset=[period_col, slot_col])
            .reset_index(drop=True)
        )

        # Remove empty string fillers, if any
        slot_df = slot_df[slot_df[slot_col].astype(str).str.strip() != ""]

        # Compute JSD
        consecutive_jsd_table = consecutive_jsd(
            temp_slot_df=slot_df,
            slot_col=slot_col,
            period_col=period_col,
            min_freq=min_freq,
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

    support_cols = [
        "slot",
        "period_1",
        "period_2",
        "support_count",
        "support_weight",
    ]
    support_key_cols = ["slot", "period_1", "period_2"]
    support_cols_set = set(support_cols)
    missing_cols = support_cols_set - set(saturating_support.columns)
    if missing_cols:
        raise ValueError(
            f"saturating_support is missing required columns: {sorted(missing_cols)}"
        )

    weighted_consecutive_jsd = consecutive_jsd.merge(
        saturating_support[support_cols],
        on=support_key_cols,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_support = weighted_consecutive_jsd["_merge"] != "both"
    if missing_support.any():
        missing_keys = weighted_consecutive_jsd.loc[
            missing_support,
            support_key_cols,
        ]
        raise ValueError(
            "Missing support for the following JSD rows:\n"
            f"{missing_keys.to_string(index=False)}"
        )

    weighted_consecutive_jsd = weighted_consecutive_jsd.drop(columns="_merge")
    weighted_consecutive_jsd["weighted_jsd"] = (
        weighted_consecutive_jsd["jsd"]
        * weighted_consecutive_jsd["support_weight"]
    )

    return weighted_consecutive_jsd[output_cols]

def compute_weighted_consecutive_jsd_df(
    sfiller_df: pd.DataFrame,
    period_col: str = "subfolder",
    min_freq: int = 1,
    mode: str = "all",
    all_periods: Optional[Sequence[Any]] = None,
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
        Minimum frequency of a filler across each compared period pair.
        Fillers below this threshold are treated as absent from that pair.

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
    _validate_period_column(sfiller_df, period_col)
    _validate_min_freq(min_freq)
    _validate_positive_k(k)
    mode = _validate_jsd_mode(mode)

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

def sfillers_jsd_by_period(
    df: pd.DataFrame,
    period_col: str = "subfolder",
    slot_col: str = "chi_amod",
    min_freq: int = 1,
    mode: str = "all",
    all_periods: Optional[Sequence[Any]] = None,
    top_n: int = 10,
    weighting: bool = False,
    k: float = 20.0,
    include_zero_slots: bool = False,
) -> Dict[Any, Dict[str, Any]]:
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
        Minimum frequency of a filler across each compared period pair.
        If a filler occurs fewer than min_freq times in the mixed
        distribution for a pair, it is treated as absent from that pair.

    mode : {"all", "data_only"}
        Period-comparison mode. ``"all"`` compares adjacent periods in
        ``all_periods``; ``"data_only"`` uses adjacent periods with raw filler
        data before pair-level filtering.

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

    df = _normalize_period_column(df, period_col)

    if all_periods is None:
        all_periods = _normalize_period_sequence(df[period_col].dropna().unique())
    else:
        all_periods = _normalize_period_sequence(all_periods, sort=False)

    jsd_df = df[[period_col, slot_col]].copy()
    jsd_df[slot_col] = jsd_df[slot_col].apply(parse_filler_cell)
    jsd_df = (
        jsd_df
        .explode(slot_col, ignore_index=True)
        .dropna(subset=[period_col, slot_col])
        .reset_index(drop=True)
    )
    jsd_df = jsd_df[jsd_df[slot_col].astype(str).str.strip() != ""]

    output = {}
    output_period_pairs = {}

    if mode == "all":
        periods = list(all_periods)
    else:
        periods = _normalize_period_sequence(jsd_df[period_col].dropna().unique())

    for period in range(1, len(periods)):
        period_1, period_2 = periods[period - 1], periods[period]

        vocab_1 = jsd_df[jsd_df[period_col] == period_1][slot_col].value_counts()
        vocab_2 = jsd_df[jsd_df[period_col] == period_2][slot_col].value_counts()

        # Apply pair-specific min_freq on the mixed distribution.
        # Fillers below min_freq across both periods are removed from this pair.
        mixed_vocab = vocab_1.add(vocab_2, fill_value=0)
        retained_fillers = mixed_vocab[mixed_vocab >= min_freq].index
        vocab_1 = vocab_1.reindex(retained_fillers, fill_value=0)
        vocab_2 = vocab_2.reindex(retained_fillers, fill_value=0)

        # Vocabulary after pair-specific filtering
        vocab = sorted(retained_fillers)

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
        output_period_pairs[period_2] = period_1

    # Apply support weight after the raw output has been created.
    if weighting:
        support_df = df[[period_col, slot_col]].copy()
        support_df[slot_col] = support_df[slot_col].apply(parse_filler_cell)

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
            period_1 = output_period_pairs[year]
            support_match = saturating_support[
                (saturating_support["slot"] == slot_col)
                & (saturating_support["period_1"].astype(str) == str(period_1))
                & (saturating_support["period_2"].astype(str) == str(year))
            ]

            if support_match.empty:
                raise ValueError(
                    "No support found for "
                    f"slot={slot_col!r}, period_1={period_1!r}, period_2={year!r}."
                )

            if len(support_match) != 1:
                raise ValueError(
                    "Expected one support row for "
                    f"slot={slot_col!r}, period_1={period_1!r}, period_2={year!r}; "
                    f"found {len(support_match)}."
                )

            support_weight = float(support_match["support_weight"].iloc[0])

            # Weight final JSD
            values["JSD"] = values["JSD"] * support_weight

            # Weight individual top filler contributions
            for item in values["top_shifted_items"]:
                item["contribution"] = item["contribution"] * support_weight

    return output

#-------------------------------------------------------------------------------
# Permutation Test Helpers
#-------------------------------------------------------------------------------
def _explode_slot_filler_rows_for_permutation(
    sfiller_df: pd.DataFrame,
    period_col: str,
) -> pd.DataFrame:
    """
    Explode slot-filler cells before period-label permutation.

    The output keeps the wide slot-column interface used by the existing JSD
    and support helpers, but each retained row contains one filler occurrence
    for one slot. This lets the permutation shuffle period labels at the
    filler-occurrence row level.
    """
    if period_col not in sfiller_df.columns:
        raise ValueError(f"Period column '{period_col}' not found in DataFrame.")

    slot_cols = [
        col for col in sfiller_df.columns
        if col not in DEFAULT_COLS and col != period_col
    ]

    if not slot_cols:
        return sfiller_df[[period_col]].copy()

    output_cols = [period_col, *slot_cols]
    exploded_frames = []

    for slot_col in slot_cols:
        slot_df = sfiller_df[[period_col, slot_col]].copy()
        slot_df[slot_col] = slot_df[slot_col].apply(parse_filler_cell)
        slot_df = (
            slot_df
            .explode(slot_col, ignore_index=True)
            .dropna(subset=[period_col, slot_col])
            .reset_index(drop=True)
        )
        slot_df = slot_df[slot_df[slot_col].astype(str).str.strip() != ""]

        if slot_df.empty:
            continue

        for other_slot_col in slot_cols:
            if other_slot_col != slot_col:
                slot_df[other_slot_col] = np.nan

        exploded_frames.append(slot_df[output_cols])

    if not exploded_frames:
        return pd.DataFrame(columns=output_cols)

    return pd.concat(exploded_frames, ignore_index=True)

def _filter_mixed_pair_frequency_for_permutation(
    exploded_df: pd.DataFrame,
    period_col: str,
    min_freq: int,
) -> pd.DataFrame:
    """
    Filter exploded slot-filler rows by mixed filler frequency per slot.

    The input must use the sparse wide layout returned by
    ``_explode_slot_filler_rows_for_permutation`` and should already be
    restricted to one period pair. A filler is retained for a slot if it occurs
    at least ``min_freq`` times across the mixed pair distribution.
    """
    _validate_min_freq(min_freq)

    if exploded_df.empty or min_freq == 1:
        return exploded_df.copy()

    slot_cols = [
        col for col in exploded_df.columns
        if col not in DEFAULT_COLS and col != period_col
    ]
    output_cols = [period_col, *slot_cols]
    filtered_frames = []

    for slot_col in slot_cols:
        slot_df = exploded_df.loc[
            exploded_df[slot_col].notna(),
            output_cols,
        ].copy()
        if slot_df.empty:
            continue

        mixed_filler_freq = slot_df.groupby(slot_col)[slot_col].transform("size")
        slot_df = slot_df[mixed_filler_freq >= min_freq]
        if not slot_df.empty:
            filtered_frames.append(slot_df)

    if not filtered_frames:
        return pd.DataFrame(columns=output_cols)

    return pd.concat(filtered_frames, ignore_index=True)

def shuffle_period_labels(
    df_pair: pd.DataFrame,
    period_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
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

def _shuffle_period_labels_within_slots(
    df_pair: pd.DataFrame,
    period_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Shuffle period labels separately for each exploded slot.

    ``df_pair`` is expected to use the sparse wide layout returned by
    ``_explode_slot_filler_rows_for_permutation``: each row contains one filler
    occurrence for exactly one slot column. Shuffling within each slot preserves
    the number of filler occurrences assigned to each period for that slot.
    """
    out = df_pair.copy()

    slot_cols = [
        col for col in out.columns
        if col not in DEFAULT_COLS and col != period_col
    ]

    for slot_col in slot_cols:
        slot_mask = out[slot_col].notna()
        if not slot_mask.any():
            continue

        labels = out.loc[slot_mask, period_col].to_numpy(copy=True)
        rng.shuffle(labels)
        out.loc[slot_mask, period_col] = labels

    return out

def chunk_list(x: Sequence[Any], chunk_size: int) -> Iterator[Sequence[Any]]:
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

def jsd_stat_df_to_keyed_values(
    jsd_df: pd.DataFrame,
    value_col: str,
) -> Dict[tuple[Any, Any, Any], float]:
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
    df_pair: pd.DataFrame,
    period_col: str,
    seeds: Sequence[int],
    min_freq: int,
    k: float,
    weighting: bool,
) -> List[Dict[tuple[Any, Any, Any], float]]:
    """
    Run a chunk of consecutive-JSD permutations.

    This worker is submitted to a process pool. ``df_pair`` is expected to
    already be exploded to one filler occurrence per row. For each seed, the
    worker shuffles period labels separately within each slot, recomputes the
    selected consecutive JSD statistic with ``mode="data_only"``, and converts
    the output to slot-transition-keyed numeric values.

    Parameters
    ----------
    df_pair : pd.DataFrame
        Exploded data restricted to one adjacent period pair.

    period_col : str
        Column containing period labels.

    seeds : sequence of int
        Random seeds for the permutations handled by this worker.

    min_freq : int
        Minimum filler frequency across each compared period pair.

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

        shuffled_df_pair = _shuffle_period_labels_within_slots(
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

def _apply_fdr_correction(result_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Yekutieli FDR correction within each slot."""
    result_df["q_value_fdr"] = np.nan
    result_df["significant_fdr_05"] = False

    for slot, idx in result_df.groupby("slot").groups.items():
        pvals = result_df.loc[idx, "p_value"].to_numpy(dtype=float)
        valid_mask = ~np.isnan(pvals)

        if valid_mask.sum() == 0:
            continue

        reject, qvals, _, _ = multipletests(
            pvals[valid_mask],
            alpha=0.05,
            method="fdr_by",
        )

        valid_indices = result_df.loc[idx].index[valid_mask]
        result_df.loc[valid_indices, "q_value_fdr"] = qvals
        result_df.loc[valid_indices, "significant_fdr_05"] = reject

    return result_df

def _empty_permutation_result_df() -> pd.DataFrame:
    """Return the standard empty permutation-test result schema."""
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
        "significant_p_value_05",
        "q_value_fdr",
        "significant_fdr_05",
    ])

def _summarize_permutation_values(
    slot_transition_key: tuple[Any, Any, Any],
    obs_value: float,
    null_values: Sequence[float],
    value_col: str,
    weighting: bool,
) -> Dict[str, Any]:
    """Summarize one observed statistic against its permutation values."""
    slot, period_1, period_2 = slot_transition_key

    arr = np.asarray(null_values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        p_value = np.nan
        null_mean = np.nan
        null_sd = np.nan
        null_q95 = np.nan
        null_q99 = np.nan
        is_significant_pval = False
    else:
        p_value = (1 + np.sum(arr >= obs_value)) / (len(arr) + 1)
        null_mean = np.mean(arr)
        null_sd = np.std(arr, ddof=1)
        null_q95 = np.quantile(arr, 0.95)
        null_q99 = np.quantile(arr, 0.99)
        is_significant_pval = p_value < 0.05

    return {
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
        "significant_p_value_05": is_significant_pval,
    }

#-------------------------------------------------------------------------------
# Permutation Test API
#-------------------------------------------------------------------------------
def permutation_test_consecutive_jsd(
    sfiller_df: pd.DataFrame,
    period_col: str = "subfolder",
    all_periods: Optional[Sequence[Any]] = None,
    n_permutations: int = 1000,
    min_freq: int = 1,
    k: float = 100,
    weighting: bool = True,
    seed: int = 42,
    keep_cols: Optional[Sequence[str]] = None,
    n_jobs: int = 8,
    chunk_size: int = 50,
) -> pd.DataFrame:
    """
    Run pairwise permutation tests for consecutive JSD.

    For each adjacent period pair in ``all_periods``, this function computes
    the observed consecutive JSD statistic for every slot column. If
    ``weighting=True``, the statistic is support-weighted JSD. If
    ``weighting=False``, the statistic is raw JSD. Before permutation, slot
    filler cells are exploded to one filler occurrence per row. The null
    distribution is built by first filtering fillers whose mixed frequency in
    the period pair is below ``min_freq``, then repeatedly shuffling period
    labels separately within each exploded slot and recomputing the same
    statistic. P-values are calculated as the proportion of null values greater
    than or equal to the observed value, with a standard plus-one correction.
    FDR correction is applied within each slot across adjacent period
    transitions.

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
        Minimum frequency of a filler across the mixed distribution of each
        compared period pair. Filtering happens before period-label
        permutation for each pair.

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

    _validate_period_column(sfiller_df, period_col)
    _validate_min_freq(min_freq)
    _validate_positive_k(k)
    _validate_permutation_arguments(n_permutations, chunk_size, n_jobs)

    master_rng = np.random.default_rng(seed)
    value_col = "weighted_jsd" if weighting else "jsd"

    if keep_cols is not None:
        required_cols = {period_col}

        keep_cols = list(set(keep_cols) | required_cols)
        sfiller_df = sfiller_df[keep_cols].copy()

    sfiller_df = _normalize_period_column(sfiller_df, period_col)

    if all_periods is None:
        all_periods = _normalize_period_sequence(sfiller_df[period_col].dropna().unique())
    else:
        all_periods = _normalize_period_sequence(all_periods, sort=False)

    results = []

    for pair_id, (p1, p2) in enumerate(zip(all_periods[:-1], all_periods[1:])):

        print(f"Permutation testing for pair {pair_id + 1}/{len(all_periods) - 1}: {p1} -> {p2}")

        df_pair = sfiller_df[sfiller_df[period_col].isin([p1, p2])].copy()

        if df_pair[period_col].nunique() < 2:
            continue

        permutation_df_pair = _explode_slot_filler_rows_for_permutation(
            sfiller_df=df_pair,
            period_col=period_col,
        )
        permutation_df_pair = _filter_mixed_pair_frequency_for_permutation(
            exploded_df=permutation_df_pair,
            period_col=period_col,
            min_freq=min_freq,
        )

        if permutation_df_pair.empty:
            continue

        # 1. Observed statistic
        if weighting:
            obs_df = compute_weighted_consecutive_jsd_df(
                sfiller_df=permutation_df_pair,
                period_col=period_col,
                min_freq=1,
                k=k,
                mode="data_only"
            )
        else:
            obs_df = compute_consecutive_jsd_df(
                sfiller_df=permutation_df_pair,
                period_col=period_col,
                min_freq=1,
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
                    permutation_df_pair,
                    period_col,
                    seed_chunk,
                    1,
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

        # 4. Summarize null distribution
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
                is_significant_pval = False
            else:
                p_value = (1 + np.sum(arr >= obs_value)) / (len(arr) + 1)
                null_mean = np.mean(arr)
                null_sd = np.std(arr, ddof=1)
                null_q95 = np.quantile(arr, 0.95)
                null_q99 = np.quantile(arr, 0.99)
                is_significant_pval = p_value < 0.05

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
                "significant_p_value_05": is_significant_pval
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
            "significant_p_value_05",
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

#-------------------------------------------------------------------------------
# Summaries
#-------------------------------------------------------------------------------
def summarize_fdr_correction(
    perm_result_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize FDR correction per slot and return fdr-corrected transitions."""
    required_cols = {
        "slot",
        "period_1",
        "period_2",
        "p_value",
        "q_value_fdr",
        "significant_p_value_05",
        "significant_fdr_05",
    }

    missing_cols = required_cols - set(perm_result_df.columns)
    if missing_cols:
        raise ValueError(
            f"perm_result_df is missing columns: {sorted(missing_cols)}"
        )

    df = perm_result_df.copy()

    # Rows that were significant before correction
    # but are no longer significant after FDR correction.
    corrected_slot_period_df = df[
        df["significant_p_value_05"]
        & ~df["significant_fdr_05"]
    ].copy()

    corrected_slot_period_df = (
        corrected_slot_period_df[
            ["slot", "period_1", "period_2", "p_value", "q_value_fdr"]
        ]
        .sort_values(["slot", "period_1", "period_2"])
        .reset_index(drop=True)
    )

    # Summarize separately for each slot because FDR correction
    # is applied independently within each slot.
    summary_rows: list[dict[str, object]] = []

    for slot, slot_df in df.groupby("slot", sort=True):
        total_tests = len(slot_df)
        valid_p = slot_df["p_value"].notna()
        valid_tests = int(valid_p.sum())
        invalid_tests = total_tests - valid_tests

        raw_significant = int(
            slot_df.loc[valid_p, "significant_p_value_05"].sum()
        )

        fdr_significant = int(
            slot_df.loc[valid_p, "significant_fdr_05"].sum()
        )

        corrected_by_fdr = raw_significant - fdr_significant

        summary_rows.append(
            {
                "slot": slot,
                "total_period_tests": total_tests,
                "valid_period_tests": valid_tests,
                "invalid_period_tests": invalid_tests,
                "raw_p_significant": raw_significant,
                "fdr_significant": fdr_significant,
                "corrected_by_fdr": corrected_by_fdr,
                "raw_significant_percent": (
                    raw_significant / valid_tests * 100
                    if valid_tests > 0
                    else np.nan
                ),
                "fdr_significant_percent": (
                    fdr_significant / valid_tests * 100
                    if valid_tests > 0
                    else np.nan
                ),
                "fdr_corrected_percent_of_raw_sig": (
                    corrected_by_fdr / raw_significant * 100
                    if raw_significant > 0
                    else 0
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return summary, corrected_slot_period_df

#-------------------------------------------------------------------------------
# Plotting And Printing
#-------------------------------------------------------------------------------
def print_jsd_by_period(jsd_results: Dict[Any, Dict[str, Any]]) -> None:
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
        for item in result["top_shifted_items"]:
            print(f"  {item['item']}: {item['contribution']:.4f}")

def plot_jsd_by_period(jsd_results: Dict[Any, Dict[str, Any]]) -> None:
    """
    Plot Jensen-Shannon Divergence values across period transitions.

    Parameters:
        jsd_results (dict): A dictionary with period as key and a dictionary as value.
            The dictionary contains the JSD and top shifted items.

    Returns:
        None
    """
    periods = list(jsd_results.keys())
    jsd_scores = [jsd_results[d]["JSD"] for d in periods]

    plt.figure(figsize=(15, 5))
    plt.plot(periods, jsd_scores, marker="o")
    plt.title("Jensen-Shannon Divergence Between Periods")
    plt.xlabel("Periods")
    plt.ylabel("JSD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_items_jsd_by_period(
    jsd_results: Dict[Any, Dict[str, Any]],
    top_n: int = 10,
    cols: int = 3,
) -> None:
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
    if not jsd_results:
        return

    if cols < 1:
        raise ValueError("cols must be >= 1.")

    num_periods = len(jsd_results)
    rows = math.ceil(num_periods / cols)

    # Find global max contribution across all periods
    global_max = max(
        max((item["contribution"] for item in result["top_shifted_items"][:top_n]), default=0)
        for result in jsd_results.values()
    )

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.atleast_1d(axes).ravel()

    for idx, (decade, result) in enumerate(jsd_results.items()):
        ax = axes[idx]
        top_words = result["top_shifted_items"][:top_n]
        labels = [
            item["item"].replace("in_", "").replace("de_", "").replace("bo_", "").replace("lo_", "")
            for item in top_words
        ]
        values = [item["contribution"] for item in top_words]

        colors = [
            "lightgreen" if item["item"].startswith("in_") else
            "lightcoral" if item["item"].startswith("de_") else
            "darkgreen" if item["item"].startswith("bo_") else
            "darkred" if item["item"].startswith("lo_") else
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
        ax.set_xlim(0, global_max * 1.05 if global_max > 0 else 1)

    # Remove unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_all_jsds_by_period(
    jsd_df: pd.DataFrame,
    slots: Optional[List[str]] = None,
    col_to_plot: Optional[str] = None,
    layout: str = "combined",
    title: str = "Weighted JSD for all slots",
    y_label: str = "JSD",
    x_label: str = "Time Period",
    height: int = 700,
    width: int = 1100,
    save_path: Optional[str] = None,
) -> go.Figure:
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

    if col_to_plot is None:
        if "weighted_jsd" in jsd_df.columns:
            col_to_plot = "weighted_jsd"
        elif "jsd" in jsd_df.columns:
            col_to_plot = "jsd"
        else:
            raise ValueError(
                "jsd_df must contain 'weighted_jsd' or 'jsd'."
            )
    elif col_to_plot not in jsd_df.columns:
        raise ValueError(
            f"col_to_plot {col_to_plot!r} not found."
            )

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

def plot_permutation_test_consecutive_jsd(
    perm_result_df: pd.DataFrame,
    y_label: Optional[str] = None,
) -> None:
    plot_df = perm_result_df.copy()

    if plot_df.empty:
        return

    if y_label is None:
        statistics = plot_df["statistic"].dropna().unique()
        if len(statistics) == 1 and statistics[0] == "weighted_jsd":
            y_label = "Weighted JSD"
        elif len(statistics) == 1 and statistics[0] == "jsd":
            y_label = "JSD"
        else:
            y_label = "Observed statistic"

    # Use period_2 as the x variable
    plot_df["period_label"] = (
        plot_df["period_1"].astype(str)
        + " vs "
        + plot_df["period_2"].astype(str)
    )

    slots = sorted(plot_df["slot"].unique())

    if not slots:
        return

    ncols = 3
    nrows = int(np.ceil(len(slots) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharex=True
    )

    axes = np.atleast_1d(axes).ravel()

    # X-axis ticks and labels
    tick_df = (
        plot_df[["period_2", "period_label"]]
        .drop_duplicates()
        .sort_values("period_2")
    )

    x_ticks = tick_df["period_2"].to_numpy()
    x_labels = tick_df["period_label"].to_numpy()

    for ax, slot in zip(axes, slots):
        s = plot_df[plot_df["slot"] == slot].sort_values("period_2")

        ax.plot(
            s["period_2"],
            s["observed_statistic"],
            marker="o",
            label=f"Observed {y_label}"
        )

        ax.plot(
            s["period_2"],
            s["null_mean"],
            linestyle="--",
            label="Null mean"
        )

        ax.plot(
            s["period_2"],
            s["null_q95"],
            linestyle=":",
            label="Null q95"
        )

        ax.plot(
            s["period_2"],
            s["null_q99"],
            linestyle="-.",
            label="Null q99"
        )

        # FDR-significant points only
        sig = s[s["significant_fdr_05"]]

        ax.scatter(
            sig["period_2"],
            sig["observed_statistic"],
            marker="*",
            s=140,
            label="FDR Sig, q < .05",
            zorder=5
        )

        ax.set_title(slot)
        ax.set_ylabel(y_label)
        ax.axhline(0, linewidth=0.8)

        # Show period labels on x-axis
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        # Because sharex=True often hides upper subplot labels
        ax.tick_params(axis="x", labelbottom=True)

    # Remove unused axes
    for ax in axes[len(slots):]:
        ax.remove()

    # Collect unique legend entries
    handles, labels = [], []
    for ax in axes[:len(slots)]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    unique = dict(zip(labels, handles))

    # Compact title + legend spacing
    fig_h = fig.get_figheight()

    title_y  = 1 - 0.08 / fig_h
    legend_y = 1 - 0.25 / fig_h
    top_axes = 1 - 0.38 / fig_h

    fig.suptitle(
        f"Observed {y_label} against permutation null distribution",
        y=title_y,
        fontsize=10
    )

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        borderaxespad=0,
        fontsize=8
    )

    fig.supxlabel("Time period", fontsize=9)

    plt.tight_layout(rect=[0, 0.035, 1, top_axes])
    plt.show()
