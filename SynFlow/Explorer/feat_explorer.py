"""Explore FEATS type distributions for target tokens across periods."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable

from SynFlow.const import DEFAULT_PATTERN


def parse_feature_types(feats: str) -> list[str]:
    """
    Extract feature type names from a FEATS field.

    FEATS is expected to use CoNLL-U style ``Type=Value|OtherType=OtherValue``.
    Empty values, ``_``, and malformed parts without ``=`` are ignored.
    """
    if not feats:
        return []

    feats = feats.strip()
    if feats == "_":
        return []

    feature_types = []
    for part in feats.split("|"):
        if "=" not in part:
            continue
        feature_type, _ = part.split("=", 1)
        feature_type = feature_type.strip()
        if feature_type:
            feature_types.append(feature_type)
    return feature_types


def process_file(args: tuple[str, str, re.Pattern[str], str, str]) -> Counter[str]:
    """
    Count FEATS type occurrences for target lemma/POS tokens in one file.

    Args:
        args: Tuple of ``(filename, corpus_folder, pattern, target_lemma, target_pos)``.
            ``pattern`` must capture token, lemma, POS, ID, HEAD, DEPREL, and FEATS.
    """
    fname, corpus_folder, pattern, target_lemma, target_pos = args
    counter: Counter[str] = Counter()
    path = os.path.join(corpus_folder, fname)

    has_target_check_string = f"\t{target_lemma}\t{target_pos}"

    with open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("<"):
                continue

            if has_target_check_string not in line:
                continue

            match = pattern.match(line)
            if not match:
                continue

            _, lemma, pos, _, _, _, feats = match.groups()
            if lemma != target_lemma or pos != target_pos:
                continue

            counter.update(parse_feature_types(feats))

    return counter


def plot_dist(counter: Counter[str], target_lemma: str, target_pos: str, subfolder: str, top_n: int) -> None:
    """Plot the top FEATS type counts for one subfolder."""
    if not counter:
        print("Nothing to plot.")
        return

    import matplotlib.pyplot as plt

    labels, freqs = zip(*counter.most_common(top_n))
    plt.figure(figsize=(min(12, max(6, 0.45 * len(labels))), 6))
    plt.bar(range(len(freqs)), freqs)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} FEATS types of {target_lemma}/{target_pos} in {subfolder}")
    plt.tight_layout()
    plt.show()


def _iter_corpus_files(subfolder_path: str) -> Iterable[str]:
    for fname in os.listdir(subfolder_path):
        if fname.endswith((".conllu", ".txt")):
            yield fname


def feat_explorer(
    corpus_folder: str,
    target_lemma: str,
    target_pos: str,
    output_folder: str,
    top_n: int = 20,
    num_processes: int | None = None,
    pattern: re.Pattern[str] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Count FEATS type distributions for target tokens in each corpus subfolder.

    The FEATS column is parsed as ``Type=Value|OtherType=OtherValue``. Counts are
    over feature types, not feature values, so ``Definite=Def|PronType=Art``
    contributes one count to ``Definite`` and one count to ``PronType``.

    Args:
        corpus_folder: Folder containing period subfolders.
        target_lemma: Lemma to match exactly.
        target_pos: POS tag to match exactly.
        output_folder: Folder where the JSON frequency file is written.
        top_n: Number of top feature types to plot per subfolder.
        num_processes: Worker count. Defaults to CPU count minus one.
        pattern: Regex capturing token, lemma, POS, ID, HEAD, DEPREL, and FEATS.
    """
    pattern = pattern or DEFAULT_PATTERN
    num_processes = num_processes or max(1, cpu_count() - 1)
    all_results: dict[str, dict[str, int]] = {}

    for subfolder in sorted(os.listdir(corpus_folder)):
        subfolder_path = os.path.join(corpus_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        files = list(_iter_corpus_files(subfolder_path))
        file_args = [
            (fname, subfolder_path, pattern, target_lemma, target_pos)
            for fname in files
        ]

        global_counter: Counter[str] = Counter()
        if num_processes == 1:
            counters = map(process_file, file_args)
        else:
            pool = Pool(num_processes)
            counters = pool.imap_unordered(process_file, file_args, chunksize=10)

        try:
            for counter in counters:
                global_counter.update(counter)
        finally:
            if num_processes != 1:
                pool.close()
                pool.join()

        print(
            f"[{subfolder}] Collected {sum(global_counter.values())} FEATS type occurrences, "
            f"{len(global_counter)} distinct types."
        )

        plot_dist(global_counter, target_lemma, target_pos, subfolder, top_n)
        all_results[subfolder] = dict(global_counter.most_common())

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_folder, f"{target_lemma}_{target_pos}_feats.json")
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=2)
    print(f"Saved FEATS type frequencies to: {output_path}")

    return all_results
