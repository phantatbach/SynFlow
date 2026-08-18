"""Explore individual and combined slot-path distributions around target tokens."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, deque
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Iterable, Sequence

from SynFlow.const import DEFAULT_PATTERN
from SynFlow.utils import build_graph


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _iter_corpus_files(subfolder_path: str) -> Iterable[str]:
    for fname in os.listdir(subfolder_path):
        if fname.endswith((".conllu", ".txt")):
            yield fname


def _target_ids(id2lemma_pos: dict[str, str], target_lemma: str, target_pos: str) -> list[str]:
    target = f"{target_lemma}/{target_pos}"
    return [idx for idx, lemma_pos in id2lemma_pos.items() if lemma_pos == target]


def _collect_counter(
    file_args: list[tuple],
    process_file: Callable[[tuple], Counter[str]],
    num_processes: int,
) -> Counter[str]:
    total: Counter[str] = Counter()
    if not file_args:
        return total

    if num_processes == 1:
        counters = map(process_file, file_args)
    else:
        pool = Pool(num_processes)
        counters = pool.imap_unordered(process_file, file_args, chunksize=10)

    try:
        for counter in counters:
            total.update(counter)
    finally:
        if num_processes != 1:
            pool.close()
            pool.join()

    return total


def _plot_counter_dist(
    counter: Counter[str],
    title: str,
    ylabel: str = "Frequency",
    top_n: int = 20,
    max_width: int = 14,
) -> None:
    if not counter:
        print("Nothing to plot.")
        return

    import matplotlib.pyplot as plt

    labels, freqs = zip(*counter.most_common(top_n))
    plt.figure(figsize=(min(max_width, max(6, 0.35 * len(labels))), 6))
    plt.bar(range(len(freqs)), freqs)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _read_target_sentences(
    path: str,
    target_lemma: str,
    target_pos: str,
) -> Iterable[list[str]]:
    has_target = False
    has_target_check_string = f"\t{target_lemma}\t{target_pos}"
    sent_tokens: list[str] = []

    with open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.rstrip("\n")

            if line.startswith("<s id"):
                sent_tokens = []
                has_target = False
            elif line.startswith("</s>"):
                if sent_tokens and has_target:
                    yield sent_tokens
            else:
                sent_tokens.append(line)
                if has_target_check_string in line:
                    has_target = True


# ---------------------------------------------------------------------------
# Individual slot-path explorer
# ---------------------------------------------------------------------------


def get_contexts(
    graph: dict[str, list[str]],
    id2deprel: dict[tuple[str, str], str],
    target_ids: list[str],
    max_length: int,
) -> list[str]:
    """
    Find all slot paths up to ``max_length`` hops from each target token.

    Paths are dependency labels joined by ``" > "``. Unlike the combination
    explorer, this returns every prefix path, so a chain ``A > B`` contributes
    both ``A`` and ``A > B`` when ``max_length`` allows it.
    """
    paths = []
    for target_id in target_ids:
        queue = deque([(target_id, 0, [], {target_id})])
        while queue:
            node, depth, path, seen = queue.popleft()
            if depth == max_length:
                continue

            for neighbor in graph.get(node, []):
                if neighbor in seen:
                    continue
                label = id2deprel.get((node, neighbor))
                if not label:
                    continue

                new_path = path + [label]
                paths.append(" > ".join(new_path))
                queue.append((neighbor, depth + 1, new_path, seen | {neighbor}))

    return paths


def _process_file_spaths(args: tuple[str, str, re.Pattern[str], str, str, int]) -> Counter[str]:
    """
    Count individual slot paths around target lemma/POS tokens in one file.

    Args:
        args: Tuple of ``(fname, corpus_folder, pattern, target_lemma,
            target_pos, max_length)``.
    """
    fname, corpus_folder, pattern, target_lemma, target_pos, max_length = args
    counter: Counter[str] = Counter()
    path = os.path.join(corpus_folder, fname)

    for sent_tokens in _read_target_sentences(path, target_lemma, target_pos):
        id2lemma_pos, graph, id2deprel = build_graph(sent_tokens, pattern)
        target_ids = _target_ids(id2lemma_pos, target_lemma, target_pos)
        for slot_path in get_contexts(graph, id2deprel, target_ids, max_length):
            counter[slot_path] += 1

    return counter


def spath_explorer(
    corpus_folder: str,
    target_lemma: str,
    target_pos: str,
    output_folder: str,
    max_length: int = 1,
    top_n: int = 20,
    num_processes: int | None = None,
    pattern: re.Pattern[str] | None = None,
) -> Counter[str]:
    """
    Count individual slot-path distributions around target tokens by subfolder.

    Writes ``{target_lemma}_{target_pos}_spaths.json`` to ``output_folder`` and
    returns the last subfolder's aggregate counter, preserving the previous API.
    """
    pattern = pattern or DEFAULT_PATTERN
    num_processes = num_processes or max(1, cpu_count() - 1)
    all_results: dict[str, dict[str, int]] = {}
    global_counter: Counter[str] = Counter()

    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        file_args = [
            (fname, subfolder_path, pattern, target_lemma, target_pos, max_length)
            for fname in _iter_corpus_files(subfolder_path)
        ]

        global_counter = _collect_counter(file_args, _process_file_spaths, num_processes)

        print(
            f"[{subfolder}] Collected {sum(global_counter.values())} context links, "
            f"{len(global_counter)} distinct arguments."
        )

        _plot_counter_dist(
            global_counter,
            title=f"Top {top_n} slot-paths of {target_lemma} (max_length={max_length})",
            top_n=top_n,
            max_width=12,
        )

        all_results[subfolder] = dict(global_counter.most_common())

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_folder, f"{target_lemma}_{target_pos}_spaths.json")
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=2)
    print(f"Saved slot-path frequencies to: {output_path}")

    return global_counter


# ---------------------------------------------------------------------------
# Slot-path combination explorer
# ---------------------------------------------------------------------------


def find_paths_from(
    graph: dict[str, list[str]],
    id2deprel: dict[tuple[str, str], str],
    start_id: str,
    max_length: int,
) -> list[str]:
    """
    Find terminal slot paths from one target token up to exactly ``max_length``.

    If a branch ends before ``max_length``, the shorter path is kept. This is
    used by ``spath_comb_explorer`` before combining unique slot paths per target
    occurrence into one atomic pattern.
    """
    paths = []

    def dfs(node: str, depth: int, seen: set[str], rel_path: list[str]) -> None:
        if depth == max_length:
            if rel_path:
                paths.append(" > ".join(rel_path))
            return

        has_neighbor = False
        for neighbor in graph.get(node, []):
            if neighbor in seen:
                continue
            label = id2deprel.get((node, neighbor))
            if not label:
                continue

            has_neighbor = True
            dfs(neighbor, depth + 1, seen | {neighbor}, rel_path + [label])

        if not has_neighbor and rel_path:
            paths.append(" > ".join(rel_path))

    dfs(start_id, 0, {start_id}, [])
    return paths


def _trim_slot_path(path: str, trimmed_rels: set[str]) -> str | None:
    """Trim one slot branch at the first excluded relation."""
    parts = [part.strip() for part in path.split(">") if part.strip()]
    kept_parts: list[str] = []

    for part in parts:
        if part in trimmed_rels:
            break
        kept_parts.append(part)

    if not kept_parts:
        return None
    return " > ".join(kept_parts)


def _process_file_spath_combs(
    args: tuple[str, str, re.Pattern[str], str, str, int, tuple[str, ...]],
) -> Counter[str]:
    """
    Count unique slot-path combinations around target lemma/POS tokens in one file.

    Args:
        args: Tuple of ``(fname, corpus_folder, pattern, target_lemma,
            target_pos, max_length, trimmed_rels)``.
    """
    fname, corpus_folder, pattern, target_lemma, target_pos, max_length, trimmed_rels = args
    trimmed_rels_set = set(trimmed_rels)
    counter: Counter[str] = Counter()
    path = os.path.join(corpus_folder, fname)

    for sent_tokens in _read_target_sentences(path, target_lemma, target_pos):
        id2lemma_pos, graph, id2deprel = build_graph(sent_tokens, pattern)
        for target_id in _target_ids(id2lemma_pos, target_lemma, target_pos):
            paths = find_paths_from(graph, id2deprel, target_id, max_length)
            if trimmed_rels_set:
                paths = [
                    trimmed_path
                    for path in paths
                    if (trimmed_path := _trim_slot_path(path, trimmed_rels_set)) is not None
                ]
            unique_paths = sorted(set(paths))
            if trimmed_rels_set and not unique_paths:
                continue
            parts = [target_lemma] + ["> " + path for path in unique_paths]
            pattern_str = " & ".join(parts)
            counter[pattern_str] += 1

    return counter


def save_to_csv_with_subfolder(rows: list[tuple[str, int, str, list[str]]], output_path: str = "output.csv") -> None:
    """
    Write combined slot-path rows as a lowercase ``&``-delimited CSV.

    Args:
        rows: Sequence of ``(subfolder, frequency, target, slots)``.
        output_path: CSV path to write.
    """
    max_slots = max((len(slots) for _, _, _, slots in rows), default=0)

    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="&")
        header = ["subfolder", "frequency", "target"] + [f"slot{i + 1}" for i in range(max_slots)]
        writer.writerow(header)

        rows_sorted = sorted(rows, key=lambda row: (row[0], -row[1], row[2]))
        for subfolder, frequency, target, slots in rows_sorted:
            row = [subfolder, frequency, target] + slots + [""] * (max_slots - len(slots))
            writer.writerow(row)

    print(f"CSV saved to {output_path}")


def spath_comb_explorer(
    corpus_folder: str,
    target_lemma: str,
    target_pos: str,
    output_folder: str,
    max_length: int = 1,
    top_n: int = 20,
    num_processes: int | None = None,
    pattern: re.Pattern[str] | None = None,
    trimmed_rels: Sequence[str] | None = None,
) -> dict[str, Counter[str]]:
    """
    Count unique slot-path combinations around target tokens by subfolder.

    Args:
        trimmed_rels: Relations that trim a single slot branch at the first
            match. If a branch starts with one of these relations, that branch is
            removed from the combination.

    Writes a lowercase ``&``-delimited CSV with ``subfolder``, ``frequency``,
    ``target``, and ``slot...`` columns.
    """
    pattern = pattern or DEFAULT_PATTERN
    num_processes = num_processes or max(1, cpu_count() - 1)
    trimmed_rels_tuple = tuple(trimmed_rels or ())
    all_totals: dict[str, Counter[str]] = {}
    csv_rows: list[tuple[str, int, str, list[str]]] = []

    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        file_args = [
            (
                fname,
                subfolder_path,
                pattern,
                target_lemma,
                target_pos,
                max_length,
                trimmed_rels_tuple,
            )
            for fname in _iter_corpus_files(subfolder_path)
        ]

        total = _collect_counter(file_args, _process_file_spath_combs, num_processes)
        all_totals[subfolder] = total

        print(f"[{subfolder}] Total instances: {sum(total.values())}, distinct patterns: {len(total)}")

        if total:
            _plot_counter_dist(
                total,
                title=(
                    f"{subfolder}: Top {top_n} unique combinations around "
                    f"{target_lemma}/{target_pos} (<={max_length}-hop)"
                ),
                ylabel="Count",
                top_n=top_n,
                max_width=14,
            )

            for pattern_str, frequency in total.items():
                parts = pattern_str.split(" & ")
                target = parts[0]
                slots = parts[1:]
                csv_rows.append((subfolder, frequency, target, slots))

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_csv = os.path.join(
        output_folder,
        f"{target_lemma}_{target_pos}_spath_combs_{max_length}_hops.csv",
    )
    save_to_csv_with_subfolder(csv_rows, output_path=output_csv)

    return all_totals
