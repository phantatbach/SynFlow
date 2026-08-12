"""Build sfiller_df-like DataFrames from token FEATS values."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Any

import pandas as pd

from SynFlow.const import DEFAULT_PATTERN

FeatureItem = tuple[str, ...]


def parse_feature_cell(feats: str) -> dict[str, list[FeatureItem]]:
    """
    Parse a FEATS field into feature-type columns and tuple-valued cells.

    FEATS is expected to use CoNLL-U style ``Type=Value|OtherType=OtherValue``.
    The returned mapping uses feature types as keys and stores feature values as
    one-element tuples, matching the atomic item format used by ``sfiller_df``.

    Examples
    --------
    ``"Definite=Def|PronType=Art"`` -> ``{"Definite": [("Def",)], "PronType": [("Art",)]}``
    ``"_"`` -> ``{}``
    """
    if not feats:
        return {}

    feats = feats.strip()
    if feats == "_":
        return {}

    parsed: dict[str, list[FeatureItem]] = defaultdict(list)
    for part in feats.split("|"):
        if "=" not in part:
            continue

        feature_type, feature_value = part.split("=", 1)
        feature_type = feature_type.strip()
        feature_value = feature_value.strip()

        if feature_type and feature_value:
            parsed[feature_type].append((feature_value,))

    return dict(parsed)


def _process_file(args: tuple[str, str, re.Pattern[str], str, str]) -> list[dict[str, Any]]:
    """
    Build feat_df rows for target lemma/POS tokens in one file.

    Args:
        args: Tuple of ``(corpus_folder, fname, pattern, target_lemma, target_pos)``.
            ``pattern`` must capture token, lemma, POS, ID, HEAD, DEPREL, and FEATS.
    """
    corpus_folder, fname, pattern, target_lemma, target_pos = args
    subfolder = os.path.basename(corpus_folder)
    path = os.path.join(corpus_folder, fname)
    rows: list[dict[str, Any]] = []

    has_target_check_string = f"\t{target_lemma}\t{target_pos}"

    with open(path, encoding="utf8") as fh:
        file_line = 0
        for line in fh:
            file_line += 1
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

            row = {
                "id": f"{target_lemma}/{fname}/{file_line}",
                "subfolder": subfolder,
                "target": [(f"{target_lemma}/{target_pos}",)],
            }
            row.update(parse_feature_cell(feats))
            rows.append(row)

    return rows


def build_feat_df(
    corpus_folder: str,
    template: str,
    target_lemma: str,
    target_pos: str,
    num_processes: int | None = None,
    pattern: re.Pattern[str] | None = None,
    output_path: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build an sfiller_df-like DataFrame from target-token FEATS values.

    ``template`` follows the same bracket syntax as ``build_sfiller_df``:
    ``"[Definite][PronType]"`` creates ``Definite`` and ``PronType`` columns.
    Feature cells use the same list-of-tuples convention as ``sfiller_df``: a
    token with ``Definite=Def`` has ``Definite == [("Def",)]``.

    Args:
        corpus_folder: Folder containing period subfolders.
        template: Bracketed feature-type template, for example
            ``"[Number][Tense][VerbForm]"``.
        target_lemma: Lemma to match exactly.
        target_pos: POS tag to match exactly.
        num_processes: Worker count. Defaults to CPU count minus one.
        pattern: Regex capturing token, lemma, POS, ID, HEAD, DEPREL, and FEATS.
        output_path: Optional CSV path to write.

    Returns:
        Tuple of ``(df, dropped)``, where ``df`` is the feature DataFrame and
        ``dropped`` contains target-token row ids with no requested feature
        values.
    """
    pattern = pattern or DEFAULT_PATTERN
    num_processes = num_processes or max(1, cpu_count() - 1)
    feature_cols = template.strip("[]").split("][") if template.strip("[]") else []
    all_rows: list[dict[str, Any]] = []

    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        fnames = [
            fname for fname in os.listdir(subfolder_path)
            if fname.endswith((".conllu", ".txt"))
        ]
        args = [
            (subfolder_path, fname, pattern, target_lemma, target_pos)
            for fname in fnames
        ]

        if num_processes == 1:
            file_results = map(_process_file, args)
        else:
            pool = Pool(num_processes)
            file_results = pool.imap_unordered(_process_file, args, chunksize=10)

        try:
            for rows in file_results:
                all_rows.extend(rows)
        finally:
            if num_processes != 1:
                pool.close()
                pool.join()

    if not all_rows:
        columns = ["id", "subfolder", "target"] + feature_cols
        df = pd.DataFrame(columns=columns).set_index("id", drop=True)
        if output_path:
            df.to_csv(output_path, encoding="utf-8")
        return df, []

    df = pd.DataFrame(all_rows).set_index("id", drop=True)
    for feature_type in feature_cols:
        if feature_type not in df:
            df[feature_type] = [[] for _ in range(len(df))]

    df = df[["subfolder", "target", *feature_cols]]
    df[feature_cols] = df[feature_cols].map(lambda value: value if isinstance(value, list) else [])

    mask = df[feature_cols].apply(lambda row: all(len(value) == 0 for value in row), axis=1)
    dropped = df.index[mask].tolist()
    df = df[~mask]

    if output_path:
        df.to_csv(output_path, encoding="utf-8")

    return df, dropped
