"""Clustering helpers for aligned slot-filler embedding vectors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize one vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize the rows of a matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def weighted_centroid(matrix: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Compute a normalized weighted centroid."""
    matrix = normalize_matrix(matrix)

    if weights is None:
        centroid = matrix.mean(axis=0)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        centroid = (matrix * weights[:, None]).sum(axis=0)

    return normalize_vector(centroid)


def agglomerative_cosine_cluster(
    matrix: np.ndarray,
    min_similarity: float = 0.45,
) -> np.ndarray:
    """Cluster vectors using average-linkage agglomerative clustering."""
    n_rows = len(matrix)
    if n_rows == 0:
        return np.array([], dtype=int)
    if n_rows == 1:
        return np.array([0], dtype=int)

    distance_threshold = 1.0 - min_similarity

    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )

    return model.fit_predict(matrix)


def get_points_for_period(
    points_df: pd.DataFrame,
    period: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return fillers, normalized vectors, and counts for one period."""
    period_df = points_df[points_df["period"] == period].copy()
    if period_df.empty:
        return [], np.empty((0, 0)), np.array([])

    words = period_df["filler"].astype(str).tolist()
    counts = period_df["count"].astype(float).to_numpy()
    matrix = np.vstack(period_df["vector"].to_numpy())
    matrix = normalize_matrix(matrix)

    return words, matrix, counts


def _rank_period_cluster_labels(
    labels: np.ndarray,
    words: list[str],
    counts: np.ndarray,
) -> dict[int, int]:
    label_stats = []
    for label in sorted(set(labels)):
        idx = np.where(labels == label)[0]
        total_count = float(counts[idx].sum())
        first_word = sorted(words[i] for i in idx)[0]
        label_stats.append((label, -total_count, first_word))

    return {
        label: period_cluster_id
        for period_cluster_id, (label, _, _) in enumerate(
            sorted(label_stats, key=lambda x: (x[1], x[2]))
        )
    }


def cluster_individual_period(
    period: int,
    words: list[str],
    matrix: np.ndarray,
    counts: np.ndarray,
    min_similarity: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster one period from scratch and return assignments plus centroids."""
    if len(words) == 0:
        return pd.DataFrame(), pd.DataFrame()

    labels = agglomerative_cosine_cluster(matrix, min_similarity=min_similarity)
    label_to_period_cluster_id = _rank_period_cluster_labels(labels, words, counts)
    assignment_rows = []
    centroid_rows = []

    for label in sorted(set(labels), key=lambda value: label_to_period_cluster_id[value]):
        idx = np.where(labels == label)[0]
        period_cluster_id = label_to_period_cluster_id[label]
        cluster_words = [words[i] for i in idx]
        cluster_counts = counts[idx]
        centroid = weighted_centroid(matrix[idx], cluster_counts)

        centroid_rows.append(
            {
                "period": period,
                "period_cluster_id": period_cluster_id,
                "cluster_id": f"{period}_{period_cluster_id}",
                "n_fillers": len(cluster_words),
                "total_count": int(cluster_counts.sum()),
                "centroid": centroid,
            }
        )

        for i in idx:
            assignment_rows.append(
                {
                    "period": period,
                    "period_cluster_id": period_cluster_id,
                    "cluster_id": f"{period}_{period_cluster_id}",
                    "filler": words[i],
                    "count": int(counts[i]),
                }
            )

    return pd.DataFrame(assignment_rows), pd.DataFrame(centroid_rows)


def run_individual_period_clustering(
    points_df: pd.DataFrame,
    min_similarity: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster every period independently."""
    assignment_dfs = []
    centroid_dfs = []

    for period in _unique_in_order(points_df["period"]):
        words, matrix, counts = get_points_for_period(points_df, period)
        period_assignments, period_centroids = cluster_individual_period(
            period=period,
            words=words,
            matrix=matrix,
            counts=counts,
            min_similarity=min_similarity,
        )
        if not period_assignments.empty:
            assignment_dfs.append(period_assignments)
            centroid_dfs.append(period_centroids)

    if not assignment_dfs:
        return pd.DataFrame(), pd.DataFrame()

    assignments_df = pd.concat(assignment_dfs, ignore_index=True)
    centroids_df = pd.concat(centroid_dfs, ignore_index=True)
    return assignments_df, centroids_df


def summarize_individual_period_clusters(assignments_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize independently estimated clusters for each period."""
    if assignments_df.empty:
        return pd.DataFrame()

    rows = []
    for (period, period_cluster_id), sub in assignments_df.groupby(
        ["period", "period_cluster_id"],
        sort=False,
    ):
        sub = sub.sort_values("count", ascending=False)
        fillers = sub["filler"].astype(str).tolist()
        rows.append(
            {
                "period": period,
                "period_cluster_id": period_cluster_id,
                "cluster_id": f"{period}_{period_cluster_id}",
                "n_fillers": len(sub),
                "total_count": int(sub["count"].sum()),
                "fillers": ", ".join(fillers),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df["_period_order"] = summary_df["period"].map(
        _period_order_map(summary_df["period"])
    )
    return (
        summary_df.sort_values(
            ["_period_order", "total_count", "period_cluster_id"],
            ascending=[True, False, True],
        )
        .drop(columns="_period_order")
        .reset_index(drop=True)
    )


def print_individual_period_clusters(assignments_df: pd.DataFrame) -> None:
    """Print independently estimated clusters grouped by period."""
    summary_df = summarize_individual_period_clusters(assignments_df)

    for period, period_df in summary_df.groupby("period", sort=False):
        print(f"\nPeriod {period}")
        for row in period_df.itertuples(index=False):
            print(
                f"  Cluster {row.period_cluster_id} "
                f"(n={row.n_fillers}, count={row.total_count}): {row.fillers}"
            )


def match_individual_clusters_between_periods(
    centroids_df: pd.DataFrame,
    period_a: int,
    period_b: int,
    top_n: int | None = 1,
) -> pd.DataFrame:
    """Compare independently estimated clusters from two periods by centroid similarity."""
    a = centroids_df[centroids_df["period"] == period_a].copy()
    b = centroids_df[centroids_df["period"] == period_b].copy()
    if a.empty or b.empty:
        return pd.DataFrame()

    matrix_a = normalize_matrix(np.vstack(a["centroid"].to_numpy()))
    matrix_b = normalize_matrix(np.vstack(b["centroid"].to_numpy()))
    similarities = cosine_similarity(matrix_a, matrix_b)

    rows = []
    for i, row_a in enumerate(a.itertuples(index=False)):
        ranked_j = np.argsort(similarities[i])[::-1]
        if top_n is not None:
            ranked_j = ranked_j[:top_n]

        for j in ranked_j:
            row_b = b.iloc[j]
            rows.append(
                {
                    "period_a": period_a,
                    "cluster_id_a": row_a.cluster_id,
                    "period_cluster_id_a": row_a.period_cluster_id,
                    "period_b": period_b,
                    "cluster_id_b": row_b["cluster_id"],
                    "period_cluster_id_b": row_b["period_cluster_id"],
                    "cosine_similarity": float(similarities[i, j]),
                }
            )

    return pd.DataFrame(rows)


@dataclass
class DiachronicCluster:
    """Incremental slot-filler cluster state."""

    cluster_id: int
    centroid: np.ndarray
    birth_period: int
    last_active_period: int
    members_by_period: dict[int, list[str]] = field(default_factory=dict)
    counts_by_period: dict[int, dict[str, int]] = field(default_factory=dict)
    centroid_by_period: dict[int, np.ndarray] = field(default_factory=dict)


def initialize_clusters(
    period: int,
    words: list[str],
    matrix: np.ndarray,
    counts: np.ndarray,
    min_similarity_new_cluster: float,
) -> tuple[dict[int, DiachronicCluster], int, pd.DataFrame]:
    """Cluster the first period from scratch."""
    clusters = {}
    next_cluster_id = 0
    assignment_rows = []
    labels = agglomerative_cosine_cluster(matrix, min_similarity=min_similarity_new_cluster)

    for label in sorted(set(labels)):
        idx = np.where(labels == label)[0]
        cluster_words = [words[i] for i in idx]
        cluster_counts = {words[i]: int(counts[i]) for i in idx}
        centroid = weighted_centroid(matrix[idx], counts[idx])
        cluster_id = next_cluster_id
        next_cluster_id += 1

        clusters[cluster_id] = DiachronicCluster(
            cluster_id=cluster_id,
            centroid=centroid,
            birth_period=period,
            last_active_period=period,
            members_by_period={period: cluster_words},
            counts_by_period={period: cluster_counts},
            centroid_by_period={period: centroid},
        )

        for word in cluster_words:
            assignment_rows.append(
                {
                    "period": period,
                    "filler": word,
                    "count": cluster_counts[word],
                    "cluster_id": cluster_id,
                    "event": "initial",
                    "similarity_to_cluster": np.nan,
                }
            )

    return clusters, next_cluster_id, pd.DataFrame(assignment_rows)


def assign_to_existing_clusters(
    words: list[str],
    matrix: np.ndarray,
    clusters: dict[int, DiachronicCluster],
    min_similarity_to_existing: float,
) -> tuple[dict[int, int], list[int], dict[int, float]]:
    """Assign words to nearest existing centroid if similarity is high enough."""
    if not clusters:
        return {}, list(range(len(words))), {}

    cluster_ids = sorted(clusters)
    centroids = np.vstack([clusters[cluster_id].centroid for cluster_id in cluster_ids])
    similarities = cosine_similarity(matrix, normalize_matrix(centroids))
    assigned = {}
    unassigned = []
    best_similarities = {}

    for i, _ in enumerate(words):
        best_j = int(np.argmax(similarities[i]))
        best_cluster_id = cluster_ids[best_j]
        best_similarity = float(similarities[i, best_j])
        best_similarities[i] = best_similarity

        if best_similarity >= min_similarity_to_existing:
            assigned[i] = best_cluster_id
        else:
            unassigned.append(i)

    return assigned, unassigned, best_similarities


def update_clusters_one_period(
    period: int,
    words: list[str],
    matrix: np.ndarray,
    counts: np.ndarray,
    clusters: dict[int, DiachronicCluster],
    next_cluster_id: int,
    min_similarity_to_existing: float,
    min_similarity_new_cluster: float,
) -> tuple[dict[int, DiachronicCluster], int, pd.DataFrame]:
    """Incrementally update clusters with one new period."""
    if len(words) == 0:
        return clusters, next_cluster_id, pd.DataFrame()

    assignment_rows = []
    assigned, unassigned, best_similarities = assign_to_existing_clusters(
        words=words,
        matrix=matrix,
        clusters=clusters,
        min_similarity_to_existing=min_similarity_to_existing,
    )

    indices_by_cluster = defaultdict(list)
    for i, cluster_id in assigned.items():
        indices_by_cluster[cluster_id].append(i)

    for cluster_id, indices in indices_by_cluster.items():
        period_words = [words[i] for i in indices]
        period_counts = {words[i]: int(counts[i]) for i in indices}
        current_centroid = weighted_centroid(matrix[indices], counts[indices])

        clusters[cluster_id].members_by_period[period] = period_words
        clusters[cluster_id].counts_by_period[period] = period_counts
        clusters[cluster_id].last_active_period = period
        clusters[cluster_id].centroid = current_centroid
        clusters[cluster_id].centroid_by_period[period] = current_centroid

        for i in indices:
            assignment_rows.append(
                {
                    "period": period,
                    "filler": words[i],
                    "count": int(counts[i]),
                    "cluster_id": cluster_id,
                    "event": "continued",
                    "similarity_to_cluster": best_similarities[i],
                }
            )

    if unassigned:
        new_rows, clusters, next_cluster_id = _create_new_clusters(
            period=period,
            words=words,
            matrix=matrix,
            counts=counts,
            unassigned=unassigned,
            clusters=clusters,
            next_cluster_id=next_cluster_id,
            min_similarity_new_cluster=min_similarity_new_cluster,
            best_similarities=best_similarities,
        )
        assignment_rows.extend(new_rows)

    return clusters, next_cluster_id, pd.DataFrame(assignment_rows)


def _create_new_clusters(
    period: int,
    words: list[str],
    matrix: np.ndarray,
    counts: np.ndarray,
    unassigned: list[int],
    clusters: dict[int, DiachronicCluster],
    next_cluster_id: int,
    min_similarity_new_cluster: float,
    best_similarities: dict[int, float],
) -> tuple[list[dict[str, object]], dict[int, DiachronicCluster], int]:
    unassigned_matrix = matrix[unassigned]
    unassigned_counts = counts[unassigned]
    unassigned_words = [words[i] for i in unassigned]
    new_labels = agglomerative_cosine_cluster(
        unassigned_matrix,
        min_similarity=min_similarity_new_cluster,
    )
    assignment_rows = []

    for label in sorted(set(new_labels)):
        local_idx = np.where(new_labels == label)[0]
        cluster_words = [unassigned_words[i] for i in local_idx]
        cluster_counts = {unassigned_words[i]: int(unassigned_counts[i]) for i in local_idx}
        centroid = weighted_centroid(unassigned_matrix[local_idx], unassigned_counts[local_idx])
        cluster_id = next_cluster_id
        next_cluster_id += 1

        clusters[cluster_id] = DiachronicCluster(
            cluster_id=cluster_id,
            centroid=centroid,
            birth_period=period,
            last_active_period=period,
            members_by_period={period: cluster_words},
            counts_by_period={period: cluster_counts},
            centroid_by_period={period: centroid},
        )

        for local_i in local_idx:
            word = unassigned_words[local_i]
            original_i = unassigned[local_i]
            assignment_rows.append(
                {
                    "period": period,
                    "filler": word,
                    "count": int(unassigned_counts[local_i]),
                    "cluster_id": cluster_id,
                    "event": "new_cluster",
                    "similarity_to_cluster": best_similarities.get(original_i, np.nan),
                }
            )

    return assignment_rows, clusters, next_cluster_id


def run_incremental_clustering(
    points_df: pd.DataFrame,
    min_similarity_to_existing: float = 0.45,
    min_similarity_new_cluster: float = 0.45,
) -> tuple[dict[int, DiachronicCluster], pd.DataFrame]:
    """Run incremental clustering over all periods in ``points_df``."""
    clusters = {}
    next_cluster_id = 0
    assignment_dfs = []

    for period in _unique_in_order(points_df["period"]):
        words, matrix, counts = get_points_for_period(points_df, period)
        if len(words) == 0:
            continue

        if not clusters:
            clusters, next_cluster_id, period_assignments = initialize_clusters(
                period=period,
                words=words,
                matrix=matrix,
                counts=counts,
                min_similarity_new_cluster=min_similarity_new_cluster,
            )
        else:
            clusters, next_cluster_id, period_assignments = update_clusters_one_period(
                period=period,
                words=words,
                matrix=matrix,
                counts=counts,
                clusters=clusters,
                next_cluster_id=next_cluster_id,
                min_similarity_to_existing=min_similarity_to_existing,
                min_similarity_new_cluster=min_similarity_new_cluster,
            )

        assignment_dfs.append(period_assignments)

    assignments_df = (
        pd.concat(assignment_dfs, ignore_index=True)
        if assignment_dfs
        else pd.DataFrame()
    )
    return clusters, assignments_df


def cluster_inspect(clusters: dict[int, DiachronicCluster]) -> None:
    """Print cluster members and counts by period for manual inspection."""
    for cluster_id, cluster in sorted(clusters.items()):
        print(f"\nCluster {cluster_id}")
        print(f"Birth: {cluster.birth_period}")
        print(f"Last active: {cluster.last_active_period}")

        for period, words in cluster.members_by_period.items():
            counts = cluster.counts_by_period.get(period, {})
            words_with_counts = [
                f"{word}({counts.get(word, 0)})"
                for word in sorted(
                    words,
                    key=lambda word: counts.get(word, 0),
                    reverse=True,
                )
            ]
            print(f"  {period}: {', '.join(words_with_counts)}")


def summarize_clusters(clusters: dict[int, DiachronicCluster]) -> pd.DataFrame:
    """Summarize incremental clusters by period."""
    rows = []

    for cluster_id, cluster in clusters.items():
        for period, words in cluster.members_by_period.items():
            counts = cluster.counts_by_period.get(period, {})
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "birth_period": cluster.birth_period,
                    "last_active_period": cluster.last_active_period,
                    "period": period,
                    "n_fillers": len(words),
                    "total_count": sum(counts.values()),
                    "fillers": ", ".join(
                        sorted(words, key=lambda word: counts.get(word, 0), reverse=True)
                    ),
                }
            )

    summary_df = pd.DataFrame(rows)
    period_order = _period_order_map(summary_df["period"])
    summary_df["_period_order"] = summary_df["period"].map(period_order)
    return (
        summary_df.sort_values(["cluster_id", "_period_order"])
        .drop(columns="_period_order")
        .reset_index(drop=True)
    )


def add_top_members_to_cluster_summary(
    cluster_summary_df: pd.DataFrame,
    assignments_df: pd.DataFrame | None = None,
    top_n: int = 3,
) -> pd.DataFrame:
    """Add a hover-members column to an incremental cluster summary."""
    summary = cluster_summary_df.copy()
    required_cols = {"period", "cluster_id", "filler", "count"}

    if (
        assignments_df is not None
        and not assignments_df.empty
        and required_cols.issubset(assignments_df.columns)
    ):
        assignments = assignments_df.copy()
        assignments["_period_order"] = assignments["period"].map(
            _period_order_map(assignments["period"])
        )
        top_members = (
            assignments
            .sort_values(
                ["_period_order", "cluster_id", "count"],
                ascending=[True, True, False],
            )
            .groupby(["period", "cluster_id"], sort=False)
            .head(top_n)
            .groupby(["period", "cluster_id"], sort=False)["filler"]
            .apply(lambda values: ", ".join(values.astype(str)))
            .reset_index(name="hover_members")
        )
        return summary.merge(top_members, on=["period", "cluster_id"], how="left")

    if "fillers" in summary.columns:
        summary["hover_members"] = (
            summary["fillers"]
            .fillna("")
            .astype(str)
            .apply(lambda text: ", ".join([item.strip() for item in text.split(",")[:top_n]]))
        )
    else:
        summary["hover_members"] = ""

    return summary


def plot_cluster_sizes_interactive(
    cluster_summary_df: pd.DataFrame,
    assignments_df: pd.DataFrame | None = None,
    title: str | None = None,
    top_n: int = 3,
) -> object:
    """Plot incremental cluster size over time with Plotly."""
    if cluster_summary_df.empty:
        raise ValueError("cluster_summary_df is empty.")

    import plotly.express as px

    plot_df = add_top_members_to_cluster_summary(
        cluster_summary_df=cluster_summary_df,
        assignments_df=assignments_df,
        top_n=top_n,
    )
    if (
        assignments_df is not None
        and not assignments_df.empty
        and "period" in assignments_df.columns
    ):
        period_order = _unique_in_order(assignments_df["period"])
    else:
        period_order = _unique_in_order(plot_df["period"])

    plot_df = plot_df.copy()
    plot_df["period"] = pd.Categorical(
        plot_df["period"],
        categories=period_order,
        ordered=True,
    )
    plot_df["cluster_label"] = plot_df["cluster_id"].astype(str)
    plot_df = plot_df.sort_values(["cluster_id", "period"])

    fig = px.line(
        plot_df,
        x="period",
        y="total_count",
        color="cluster_label",
        line_group="cluster_label",
        markers=True,
        custom_data=["cluster_id", "total_count", "hover_members", "n_fillers", "birth_period"],
        title=title or "Incremental filler clusters over time",
        category_orders={"period": period_order},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Cluster %{customdata[0]}</b><br>"
            "Period: %{x}<br>"
            "Total count: %{customdata[1]}<br>"
            "Number of fillers: %{customdata[3]}<br>"
            "Birth period: %{customdata[4]}<br>"
            "Top members: %{customdata[2]}"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        xaxis_title="Period",
        yaxis_title="Total filler count",
        legend_title="Cluster",
        hovermode="closest",
        width=1000,
        height=600,
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=period_order,
    )
    fig.show()
    return fig


def _unique_in_order(periods: pd.Series) -> list[object]:
    return periods.dropna().drop_duplicates().tolist()


def _period_order_map(periods: pd.Series) -> dict[object, int]:
    return {period: index for index, period in enumerate(_unique_in_order(periods))}


def detect_reassignments(assignments_df: pd.DataFrame) -> pd.DataFrame:
    """Return fillers whose assigned incremental cluster changes over time."""
    rows = []
    assignments = assignments_df.copy()
    assignments["_period_order"] = assignments["period"].map(
        _period_order_map(assignments["period"])
    )

    for filler, sub in assignments.groupby("filler"):
        sub = sub.sort_values("_period_order")
        previous_period = None
        previous_cluster = None

        for row in sub.itertuples(index=False):
            if previous_cluster is not None and row.cluster_id != previous_cluster:
                rows.append(
                    {
                        "filler": filler,
                        "from_period": previous_period,
                        "to_period": row.period,
                        "from_cluster": previous_cluster,
                        "to_cluster": row.cluster_id,
                    }
                )

            previous_period = row.period
            previous_cluster = row.cluster_id

    return pd.DataFrame(rows)
