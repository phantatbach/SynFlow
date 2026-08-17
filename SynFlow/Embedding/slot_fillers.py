"""Slot-filler selection and period-local embedding plots."""

from __future__ import annotations

import ast
import math

import numpy as np
import pandas as pd

from .histwords import HistWordsSlice


def parse_slot_values(value: object) -> list[str]:
    """
    Parse one slot-filler cell into single-node filler strings.

    Embedding plots only support atomic tuple fillers with exactly one element.
    Multi-depth tuple fillers raise ValueError instead of being flattened or
    parsed as tuple strings.
    """
    if value is None or value is pd.NA:
        return []

    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, tuple):
        raw_values = [value]
    else:
        if isinstance(value, float) and math.isnan(value):
            return []
        try:
            raw_values = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return []
        if isinstance(raw_values, tuple):
            raw_values = [raw_values]
        elif not isinstance(raw_values, list):
            raw_values = [raw_values]

    fillers = []
    for raw_filler in raw_values:
        if isinstance(raw_filler, list):
            raw_filler = tuple(raw_filler)
        if isinstance(raw_filler, tuple):
            if len(raw_filler) != 1:
                raise ValueError(
                    "Embedding workflow only supports tuple fillers with exactly one element."
                )
            raw_filler = raw_filler[0]

        filler = str(raw_filler).strip()
        if not filler:
            continue
        if "/" in filler:
            filler = filler.rsplit("/", 1)[0]
        if filler:
            fillers.append(filler)

    return fillers


def collect_slot_fillers_by_period(
    slot_df: pd.DataFrame,
    slot_col: str,
    period_col: str = "subfolder",
) -> pd.DataFrame:
    """Count fillers for one slot within each period."""
    if slot_col not in slot_df.columns:
        raise KeyError(f"Missing slot column: {slot_col}")
    if period_col not in slot_df.columns:
        raise KeyError(f"Missing period column: {period_col}")

    rows = []
    for period_value, slot_value in zip(slot_df[period_col], slot_df[slot_col]):
        period = _normalize_period_label(period_value)
        if period is None:
            continue
        for filler in parse_slot_values(slot_value):
            rows.append(
                {
                    "period": period,
                    "slot": f"{slot_col}_{period}",
                    "filler": filler,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["period", "slot", "filler", "count"])

    rows_df = pd.DataFrame(rows)
    period_order = _period_order_map(rows_df["period"])
    freq_df = (
        rows_df.groupby(["period", "slot", "filler"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    freq_df["_period_order"] = freq_df["period"].map(period_order)
    return (
        freq_df.sort_values(
            ["_period_order", "count", "filler"],
            ascending=[True, False, True],
        )
        .drop(columns="_period_order")
        .reset_index(drop=True)
    )


def select_slot_fillers(
    filler_freq_df: pd.DataFrame,
    min_freq: int | None = None,
    top_n: int | None = None,
    periods: list[str | int] | None = None,
) -> pd.DataFrame:
    """Filter slot fillers by period frequency and optional per-period rank."""
    selected_df = filler_freq_df.copy()
    if periods is not None:
        period_labels = [
            period
            for period in (_normalize_period_label(value) for value in periods)
            if period is not None
        ]
        selected_df = selected_df[selected_df["period"].isin(period_labels)]
        period_order = {period: index for index, period in enumerate(period_labels)}
    else:
        period_order = _period_order_map(selected_df["period"])

    if min_freq is not None:
        selected_df = selected_df[selected_df["count"] >= min_freq]

    selected_df["_period_order"] = selected_df["period"].map(period_order)
    selected_df = selected_df.sort_values(
        ["_period_order", "count", "filler"],
        ascending=[True, False, True],
    )
    if top_n is None:
        return selected_df.drop(columns="_period_order").reset_index(drop=True)

    return (
        selected_df.groupby("period", sort=False, group_keys=False)
        .head(top_n)
        .drop(columns="_period_order")
        .reset_index(drop=True)
    )


def build_slot_embedding_points(
    selected_fillers_df: pd.DataFrame,
    embeddings: dict[str | int, HistWordsSlice],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach period-specific embedding vectors to selected fillers."""
    rows = []
    missing_rows = []

    for row in selected_fillers_df.itertuples(index=False):
        period = row.period
        embedding = _get_period_embedding(embeddings, period)
        if embedding is None or not embedding.has_word(row.filler):
            missing_rows.append(
                {
                    "period": period,
                    "slot": row.slot,
                    "filler": row.filler,
                    "count": int(row.count),
                }
            )
            continue

        rows.append(
            {
                "period": period,
                "slot": row.slot,
                "filler": row.filler,
                "count": int(row.count),
                "vector": embedding.vector(row.filler),
            }
        )

    points_df = pd.DataFrame(rows, columns=["period", "slot", "filler", "count", "vector"])
    missing_df = pd.DataFrame(missing_rows, columns=["period", "slot", "filler", "count"])
    return points_df, missing_df


def _normalize_period_label(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    period = str(value).strip()
    return period or None


def _period_order_map(periods: pd.Series) -> dict[object, int]:
    return {period: index for index, period in enumerate(pd.unique(periods.dropna()))}


def _get_period_embedding(
    embeddings: dict[str | int, HistWordsSlice],
    period: str | int,
) -> HistWordsSlice | None:
    if period in embeddings:
        return embeddings[period]

    period_label = _normalize_period_label(period)
    if period_label is not None and period_label in embeddings:
        return embeddings[period_label]

    if period_label is not None and period_label.isdigit():
        period_number = int(period_label)
        if period_number in embeddings:
            return embeddings[period_number]

    return None


def add_period_pca_coordinates(points_df: pd.DataFrame) -> pd.DataFrame:
    """Project fillers separately within each period embedding space."""
    period_dfs = []
    for _, period_df in points_df.groupby("period"):
        if len(period_df) < 2:
            continue

        vectors = np.vstack(period_df["vector"].to_numpy())
        centered_vectors = vectors - vectors.mean(axis=0, keepdims=True)
        _, _, components = np.linalg.svd(centered_vectors, full_matrices=False)
        coordinates = centered_vectors @ components[:2].T

        result_df = period_df.drop(columns=["vector"]).copy()
        result_df["x"] = coordinates[:, 0]
        result_df["y"] = coordinates[:, 1] if coordinates.shape[1] > 1 else 0.0
        period_dfs.append(result_df)

    if not period_dfs:
        return pd.DataFrame(columns=["period", "slot", "filler", "count", "x", "y"])

    return pd.concat(period_dfs, ignore_index=True)


def scale_marker_sizes(
    counts: pd.Series,
    min_size: float = 40.0,
    max_size: float = 350.0,
) -> pd.Series:
    """Scale filler frequencies to scatter marker areas."""
    if counts.empty:
        return counts.astype(float)

    min_count = counts.min()
    max_count = counts.max()
    if min_count == max_count:
        return pd.Series([(min_size + max_size) / 2] * len(counts), index=counts.index)

    return min_size + (counts - min_count) * (max_size - min_size) / (max_count - min_count)


def plot_slot_fillers_by_period(
    points_df: pd.DataFrame,
    slot_col: str,
    ncols: int = 4,
    min_point_size: float = 40.0,
    max_point_size: float = 350.0,
) -> None:
    """Plot one PCA scatter plot per period."""
    import matplotlib.pyplot as plt

    period_values = points_df["period"].dropna().drop_duplicates().tolist()
    if not period_values:
        raise ValueError("No in-vocabulary fillers to plot.")

    nrows = math.ceil(len(period_values) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)

    for ax, period in zip(axes.flat, period_values):
        period_df = points_df[points_df["period"] == period]
        marker_sizes = scale_marker_sizes(
            period_df["count"],
            min_size=min_point_size,
            max_size=max_point_size,
        )
        ax.scatter(period_df["x"], period_df["y"], s=marker_sizes, alpha=0.75)

        for row in period_df.itertuples(index=False):
            ax.annotate(row.filler, (row.x, row.y), fontsize=8, alpha=0.85)

        ax.axhline(0, color="0.85", linewidth=1)
        ax.axvline(0, color="0.85", linewidth=1)
        ax.set_title(f"{slot_col}_{period}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    for ax in axes.flat[len(period_values) :]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()
