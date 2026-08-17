from __future__ import annotations

import os
import re
from ast import literal_eval
from multiprocessing import Pool, cpu_count
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from SynFlow.const import DEFAULT_COLS, DEFAULT_PATTERN, VALID_FILLER_FORMATS
from SynFlow.utils import build_graph, format_filler

FillerItem = tuple[str, ...]

#-------------------------------------------------------------------------------
# Construction Helpers
#-------------------------------------------------------------------------------
def reformat_deprel(label: str) -> str:
    """Strip 'chi_' or 'pa_' prefixes from a dependency label."""
    return re.sub(r"^(chi_|pa_)", "", label)

def follow_path(graph, id2deprel, start, rel_seq):
    """
    Follows a path specified by rel_seq from start in graph.
    
    Args:
        graph (dict): Dependency graph mapping each token id to its neighbors.
        id2deprel (dict): Mapping of edge (tuple of token ids) to dependency relation label.
        start (int): The id of the starting node.
        rel_seq (list[str]): The sequence of dependency labels to follow.
    
    Returns:
        list[list[int]]: A list of paths, where each path is a list of node ids.
    """
    chains = []
    def dfs(node, i, path_nodes):
        """
        Recursively follows a path specified by rel_seq from node in graph.
        
        Args:
            node (int): The id of the current node.
            i (int): The index in rel_seq we're currently at.
            path_nodes (list[int]): The list of node ids we've seen so far.
        
        Returns:
            None
        """
        if i == len(rel_seq): # if index = len(rel_seq), we've reached the end
            chains.append(path_nodes) # append all nodes in the path
            return # End the current path
        expected_rel = rel_seq[i]
        for nb in graph[node]:
            if id2deprel.get((node, nb)) == expected_rel: # Check if the edge label is the expected_rel
                dfs(nb, i+1, path_nodes + [nb])
    dfs(start, 0, [])
    return chains

def process_file(args) -> list[dict[str, Any]]:
    corpus_folder, fname, pattern, target_lemma, target_pos, slots, filtered_pos, filler_format = args # Use this for multiprocess.Pool
    pattern = pattern or DEFAULT_PATTERN
    
    subfolder = os.path.basename(corpus_folder)  # <— tên subfolder
    filtered_pos = filtered_pos or [] # Guard if filtered_pos is None
    out = []
    path = os.path.join(corpus_folder, fname)

    has_target = False
    has_target_check_string = f"\t{target_lemma}\t{target_pos}"

    with open(path, encoding="utf8") as fh:
        file_line = 0
        sent_tokens, sent_lines = [], [] # Init for the whole file. Sent_tokens = lines, sent_forms = word forms only

        for line in fh:
            file_line += 1
            line = line.rstrip("\n")

            # Start a new sentence
            if line.startswith("<s id"):
                sent_tokens, sent_lines = [], [] # Reset for new sentence
                has_target = False # Reset for new sentence

            # End of a sentence. Build graph and process if target found
            elif line.startswith("</s>"):
                if sent_tokens and has_target == True:
                    # Build graph when the whole sentence is appended
                    id2lemma_pos, graph, id2deprel = build_graph(sent_tokens, pattern)
                    target_lp = f"{target_lemma}/{target_pos}"
                    for tid, lp in id2lemma_pos.items():
                        if lp != target_lp: # Only process the matched token
                            continue
                        token_line = sent_lines[int(tid)-1]
                        row = {
                            "id": f"{target_lemma}/{fname}/{token_line}",
                            "subfolder": subfolder,
                            }

                        for slot in slots:
                            slot_fillers: list[FillerItem] = []
                            # split if there are multiple fillers in a slot
                            for subslot in slot.split("|"):
                                # split your multi-hop slot
                                rel_seq = [r.strip() for r in subslot.split(">")]
                                # get every chain of IDs matching that rel sequence
                                chains  = follow_path(graph, id2deprel, tid, rel_seq)
                                # print(f"DEBUG {fname}:{token_line} chains for {slot} =", chains)

                                for chain in chains:
                                    prev_id = tid
                                    chain_fillers = []
                                    for nid in chain:
                                        lemma_pos = id2lemma_pos[nid]
                                        lemma, pos = lemma_pos.rsplit("/", 1)
                                        if pos in filtered_pos:
                                            prev_id = nid
                                            continue

                                        orig_line = sent_tokens[int(nid)-1]
                                        m = pattern.match(orig_line)
                                        token = m.group(1) if m else lemma
                                        deprel = (
                                            id2deprel.get((prev_id, nid), "UNK")
                                            if filler_format.endswith("/deprel")
                                            else None
                                        )
                                        filler = format_filler(token, lemma, pos, deprel, filler_format)

                                        chain_fillers.append(filler)
                                        prev_id = nid

                                    if chain_fillers:
                                        slot_fillers.append(tuple(chain_fillers))

                            row[slot] = slot_fillers

                        out.append(row)
            else:
                sent_tokens.append(line)
                sent_lines.append(file_line)
                # Check for target lemma/POS in the current line
                if has_target_check_string in line:
                    has_target = True

    return out

def get_all_slots(df: pd.DataFrame) -> str:
    all_slots = "".join(f"[{r}]" for r in df.index)
    return all_slots

def build_sfiller_df(
    corpus_folder: str,
    template: str,
    target_lemma: str,
    target_pos: str,
    filler_format: str = "lemma/pos",
    num_processes: int | None = None,
    pattern: re.Pattern | None = None,
    filtered_pos: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a slot-filler DataFrame from a parsed corpus.

    Each matched dependency chain is stored as one atomic tuple. Single-depth
    matches are normalized to one-element tuples, and multi-depth matches
    preserve path order inside the tuple. Multiple subslots separated by ``|``
    contribute separate tuple items to the same slot column.

    When saved with pandas, these cells are serialized with Python ``repr``
    syntax, for example ``"[('bark/NOUN', 'the/DET')]"``. The parser helpers in
    this module expect that Python-literal list-of-tuples format when reading
    CSV output back in.

    Returns:
        Tuple of ``(df, dropped)``, where ``df`` is the slot-filler DataFrame and
        ``dropped`` contains row ids where all requested slots were empty.
    """
    pattern   = pattern or DEFAULT_PATTERN
    num_procs = num_processes or max(1, cpu_count()-1)
    slots     = template.strip("[]").split("][")
    filtered_pos = filtered_pos or [] # Guard if filtered_POS is None
    filler_format = filler_format or "lemma/pos"
    if filler_format not in VALID_FILLER_FORMATS:
        valid_formats = ", ".join(sorted(VALID_FILLER_FORMATS))
        raise ValueError(f"filler_format must be one of: {valid_formats}")
    
    all_rows = []
    skipped_non_subfolders: list[str] = []

    # Go through each subfolder in the corpus folder
    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            skipped_non_subfolders.append(subfolder_path)
            continue

        fnames    = [f for f in os.listdir(subfolder_path)
                if f.endswith((".conllu", ".txt"))]
        
        args = [
            (subfolder_path, f, pattern, target_lemma, target_pos, slots, filtered_pos, filler_format)
            for f in fnames
        ]
    

        # Parallel file processing
        with Pool(num_procs) as pool:
            for rows in pool.imap_unordered(process_file, args, chunksize=10):
                all_rows.extend(rows)

    if skipped_non_subfolders:
        print(f"Skipped non-subfolder entries: {skipped_non_subfolders}")

    # Build DataFrame   
    df = pd.DataFrame(all_rows).set_index("id", drop=True)

    # ensure each slot column exists, even empty columns
    for slot in slots:
        if slot not in df:
            df[slot] = [[]] * len(df)

    # drop empty‐slot rows
    mask = df[slots].apply(lambda r: all(len(x)==0 for x in r), axis=1)
    dropped = df.index[mask].tolist()
    df = df[~mask]

    # --- Optional: insert the new "target" slot at column 0 ------------
    target_slot = f"{target_lemma}/{target_pos}"
    # Create a column of single‐item lists [target_slot] for every row:
    df.insert(1, "target", [[(target_slot,)] for _ in range(len(df))])

    return df, dropped


def _is_missing_value(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, (float, np.floating)) and np.isnan(value)


def _parse_list_cell(cell: Any) -> list[Any]:
    if isinstance(cell, list):
        return cell

    if isinstance(cell, tuple):
        return [cell]

    if _is_missing_value(cell):
        return []

    if isinstance(cell, str):
        stripped = cell.strip()
        if stripped in {"", "[]", "nan", "None", "<NA>"}:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = literal_eval(stripped)
            except (SyntaxError, ValueError):
                return [cell]
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple):
                return [parsed]
            if _is_missing_value(parsed):
                return []
            return [parsed]
        return [cell]

    return [cell]


def _normalize_filler_item(item: Any) -> FillerItem | None:
    if _is_missing_value(item):
        return None

    if isinstance(item, tuple):
        values = item
    elif isinstance(item, list):
        values = tuple(item)
    else:
        values = (item,)

    normalized_values = []
    for value in values:
        if _is_missing_value(value):
            continue
        normalized_value = str(value).strip()
        if normalized_value not in {"", "nan", "None", "<NA>"}:
            normalized_values.append(normalized_value)

    return tuple(normalized_values) if normalized_values else None


def _cell_to_filler_items(cell: Any) -> list[FillerItem]:
    output = []
    for item in _parse_list_cell(cell):
        normalized_item = _normalize_filler_item(item)
        if normalized_item is not None:
            output.append(normalized_item)
    return output


#-------------------------------------------------------------------------------
# Column Editing
#-------------------------------------------------------------------------------
def replace_in_sfiller_df_column(
    sfiller_df_path: str,
    column_name: str,
    replacements: Mapping[str, str],
    output_path: str,
) -> None:
    """
    Replace slot-filler values in one CSV column and write the updated CSV.

    The target column is expected to contain string representations of Python
    lists of tuples, such as ``"[('big/A',), ('open/A',)]"`` or
    ``"[('bark/NOUN', 'the/DET')]"``. Each tuple element is looked up in
    ``replacements``; matching elements are replaced with their mapped value,
    and unmatched elements are kept unchanged.

    Args:
        sfiller_df_path (str): Path to the input slot-filler DataFrame.
        column_name (str): Name of the column whose list values should be
            rewritten.
        replacements (dict): Mapping from original filler values to replacement
            values.
        output_path (str): Path where the updated CSV should be saved.
    """
    sfiller_df = pd.read_csv(sfiller_df_path, encoding="utf-8")

    def replace_list_str(cell: Any) -> str:
        items = _cell_to_filler_items(cell)
        replaced_items = [
            tuple(replacements.get(element, element) for element in item)
            for item in items
        ]
        return str(replaced_items)

    sfiller_df[column_name] = sfiller_df[column_name].astype(str).map(replace_list_str)

    sfiller_df.to_csv(output_path, index=False, encoding="utf-8")

def merge_sfiller_df_columns(
    sfiller_df_path: str,
    merge_formula: Mapping[str, Sequence[str]] | Sequence[tuple[str, Sequence[str]]] | Sequence[dict],
    output_path: str | None = None,
    drop_source_columns: bool = True,
    deduplicate: bool = False,
) -> pd.DataFrame:
    """
    Merge tuple-valued slot-filler columns and optionally remove source columns.

    Every cell is normalized to ``list[tuple[str, ...]]`` before merging. If
    ``deduplicate`` is true, duplicate atomic tuples inside each merged cell are
    removed while preserving first-seen order.

    Args:
        sfiller_df_path (str): Path to the input slot-filler CSV.
        merge_formula: Column merge specification. The simplest form is a dict:
            ``{"new_column": ["old_col_1", "old_col_2"]}``.
            It also accepts ``[("new_column", ["old_col_1", "old_col_2"])]`` or
            ``[{"output": "new_column", "columns": ["old_col_1", "old_col_2"]}]``.
        output_path (str | None): Where to save the merged CSV. If ``None``, the
            DataFrame is returned without writing a file.
        drop_source_columns (bool): If True, delete columns used for merging.
            When the output column is also a source column, it is kept.
        deduplicate (bool): If True, remove duplicate fillers inside each merged
            cell while preserving their first-seen order.

    Returns:
        pd.DataFrame: The merged slot-filler DataFrame.
    """
    sfiller_df = pd.read_csv(sfiller_df_path, encoding="utf-8")
    all_missing_cols = []

    def normalize_formula(formula):
        if isinstance(formula, Mapping):
            return list(formula.items())

        normalized = []
        for spec in formula:
            if isinstance(spec, Mapping):
                output_col = spec.get("output") or spec.get("new_column") or spec.get("target")
                source_cols = spec.get("columns") or spec.get("source_columns") or spec.get("sources")
                if output_col is None or source_cols is None:
                    raise ValueError(
                        "Merge specs given as dicts must contain an output/new_column/target "
                        "and columns/source_columns/sources."
                    )
                normalized.append((output_col, source_cols))
            else:
                output_col, source_cols = spec
                normalized.append((output_col, source_cols))
        return normalized

    def normalize_source_cols(source_cols):
        if isinstance(source_cols, str):
            return [source_cols]
        return list(source_cols)

    def merge_row(row, source_cols):
        merged = []
        seen = set()
        for source_col in source_cols:
            for item in _cell_to_filler_items(row[source_col]):
                if deduplicate:
                    key = repr(item)
                    if key in seen:
                        continue
                    seen.add(key)
                merged.append(item)
        return merged
    normalized_formula = normalize_formula(merge_formula)

    formula_source_cols = set()

    for output_col, source_cols in normalized_formula:
        source_cols = normalize_source_cols(source_cols)
        formula_source_cols.update(source_cols)

    df_cols = set(sfiller_df.columns)

    df_cols_not_in_formula = sorted(df_cols - formula_source_cols - set(DEFAULT_COLS))

    if df_cols_not_in_formula:
        print("Columns in DataFrame but not in merge_formula:")
        print(df_cols_not_in_formula)

    for output_col, source_cols in normalized_formula:
        source_cols = normalize_source_cols(source_cols)
        missing_cols = [col for col in source_cols if col not in sfiller_df.columns]
        existing_source_cols = [col for col in source_cols if col in sfiller_df.columns]

        all_missing_cols.extend(missing_cols)

        if not existing_source_cols:
            sfiller_df[output_col] = [[] for _ in range(len(sfiller_df))]
            continue

        insert_at = min(sfiller_df.columns.get_loc(col) for col in existing_source_cols)

        merged_values = sfiller_df.apply(
            lambda row: merge_row(row, existing_source_cols),
            axis=1
        )

        if output_col in sfiller_df.columns:
            sfiller_df[output_col] = merged_values
        else:
            sfiller_df.insert(insert_at, output_col, merged_values)

        if drop_source_columns:
            cols_to_drop = [col for col in existing_source_cols if col != output_col]
            sfiller_df = sfiller_df.drop(columns=cols_to_drop)

    if all_missing_cols:
        print("Columns in merge_formula but not in DataFrame:")
        print(sorted(set(all_missing_cols)))

    if output_path:
        sfiller_df.to_csv(output_path, index=False, encoding="utf-8")

    return sfiller_df

#-------------------------------------------------------------------------------
# Slot Extraction
#-------------------------------------------------------------------------------
def _non_empty(v: Any) -> bool:
    return len(_cell_to_filler_items(v)) > 0

def extract_slot_cols(
    spath_df: str,
    slot_names: Sequence[str],
    output_path: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(spath_df)
    slot_names = list(slot_names)
    cols = [c for c in DEFAULT_COLS + slot_names if c in df.columns]
    sub = df[cols].copy()
    keep = sub[slot_names].map(_non_empty).any(axis=1)
    sub = sub[keep]
    if output_path:
        sub.to_csv(output_path, index=False)
    return sub

def explode_slot_col(df: pd.DataFrame, slot_name: str) -> pd.DataFrame:
    """
    Explode a slot column so each row contains one atomic filler tuple.

    The exploded filler is wrapped back into a one-item list to preserve the
    list-of-tuples CSV format used by downstream parsers.
    """
    if slot_name not in df.columns:
        raise KeyError(f"Column not found: {slot_name}")

    exploded_df = df.copy()
    exploded_df[slot_name] = exploded_df[slot_name].apply(parse_filler_cell)
    exploded_df = exploded_df.explode(slot_name, ignore_index=True)
    exploded_df = exploded_df[exploded_df[slot_name].notna()].reset_index(drop=True)
    exploded_df[slot_name] = exploded_df[slot_name].map(lambda item: [item])
    return exploded_df

def extract_1_slot_col(
    spath_df: str,
    slot_name: str,
    output_path: str | None = None,
    explode: bool = False,
) -> pd.DataFrame:
    slot_col_df = extract_slot_cols(spath_df, [slot_name])

    if explode:
        slot_col_df = explode_slot_col(slot_col_df, slot_name)

    if output_path:
        slot_col_df.to_csv(output_path, index=False)

    return slot_col_df

#-------------------------------------------------------------------------------
# Parsing And Counting Helpers
#-------------------------------------------------------------------------------
def _count_fillers(cell: Any) -> int:
    """
    Count how many fillers are present in one CSV cell.

    Examples:
        "[]" -> 0
        "[('the',)]" -> 1
        "[('white',), ('powerful',)]" -> 2
        "[('bark', 'the')]" -> 1
    """
    return len(_cell_to_filler_items(cell))

def _normalize_period_sequence(periods) -> list[str]:
    """Convert period labels to strings while preserving input order."""
    normalized_periods = [
        str(period)
        for period in periods
        if not pd.isna(period)
    ]
    return normalized_periods

def _normalize_period_column(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """Return a copy with non-missing period labels converted to strings."""
    out = df.copy()
    out[period_col] = out[period_col].astype("object")
    period_mask = out[period_col].notna()
    out.loc[period_mask, period_col] = out.loc[period_mask, period_col].map(str)
    return out

def parse_filler_cell(cell: Any) -> list[FillerItem]:
    """
    Parse one slot-filler cell and return a list of atomic filler tuples.

    The function is idempotent: applying it multiple times produces the same
    result. Scalar strings are normalized to one-element tuples.

    CSV cells must use Python-literal list-of-tuples syntax when representing
    structured fillers, for example ``"[('bark/NOUN', 'the/DET')]"``.

    Examples
    --------
    "[('on',), ('in',)]"        -> [("on",), ("in",)]
    [("on",), ("in",)]          -> [("on",), ("in",)]
    [("bark", "the")]           -> [("bark", "the")]
    ["legacy", "flat"]          -> [("legacy",), ("flat",)]
    "200"                       -> [("200",)]
    NaN                         -> []
    """
    return _cell_to_filler_items(cell)

#-------------------------------------------------------------------------------
# Support Weighting
#-------------------------------------------------------------------------------
def compute_saturating_support_from_sfiller_df(
    sfiller_df: pd.DataFrame,
    period_col: str = "subfolder",
    min_freq: int = 1,
    mode: str = "all",
    all_periods=None,
    k: float = 30.0,
    include_zero_slots: bool = False,
) -> pd.DataFrame:
    """
    Process a slot-filler DataFrame to calculate saturating support for
    each slot between consecutive periods.

    The support count is calculated after applying a minimum atomic filler
    frequency threshold to the mixed distribution of each compared period pair.

    Example:
        If min_freq = 2 and atomic filler ``("b",)`` occurs once in period A
        and once in period B, then ``("b",)`` is kept for the A-B comparison.
        If ``("b",)`` occurs once in period A and zero times in period B, then
        ``("b",)`` is ignored for the A-B comparison.

    Steps:
    1. Read a DataFrame where each slot column contains atomic filler tuples or
       cells that parse to lists of atomic filler tuples.
    2. For each slot and adjacent period pair, count atomic filler tuple
       frequencies across the mixed pair distribution.
    3. Remove atomic filler tuples whose mixed frequency is < min_freq for that pair.
    4. Aggregate remaining filler counts into raw slot counts by period.
    5. For each consecutive period pair, compute:
           count_support(slot, t-t+1) = min(raw_count(slot, t), raw_count(slot, t+1))
    6. Convert count support into a bounded saturating weight:
           weight = min(1, c / k)

    Parameters
    ----------
    sfiller_df:
        Slot-filler DataFrame.

    period_col:
        Column containing the period/bin information.
        Default: "subfolder".

    min_freq:
        Minimum frequency of a filler across each compared period pair.
        Fillers below this frequency are treated as absent from that pair.
        Default: 1.

    mode:
        Period-comparison mode.
        ``"all"`` compares adjacent periods in the complete dataset timeline.
        ``"data_only"`` compares adjacent periods with raw filler data for each
        slot, then skips pairs without data on both sides after mixed filtering.

    all_periods:
        Complete period sequence for ``mode="all"``. If None, periods are
        inferred from the DataFrame. Ignored when ``mode="data_only"``.

    k:
        Support threshold.
        If c >= k, then weight = 1.0.
        Larger k penalizes low counts more strongly.
        Default: 30.0.

    include_zero_slots:
        If False, only return slots that occur at least once after filtering.
        If True, return all slot columns, including those with only zero counts.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``slot``, ``period_1``, ``period_2``,
        ``support_count``, and ``support_weight``.
    """

    if k <= 0:
        raise ValueError("`k` must be > 0.")

    if min_freq < 1:
        raise ValueError("`min_freq` must be >= 1.")

    mode = mode.lower()
    if mode not in {"all", "data_only"}:
        raise ValueError(
            f"`mode` must be either 'all' or 'data_only', but got {mode!r}."
        )

    df = sfiller_df.copy()

    if period_col not in df.columns:
        raise ValueError(f"Period column '{period_col}' not found in DataFrame.")

    df = _normalize_period_column(df, period_col)

    # Slot columns are all columns except metadata columns.
    # Also explicitly exclude period_col for safety.
    slot_cols = [
        col for col in df.columns
        if col not in DEFAULT_COLS and col != period_col
    ]

    if not slot_cols:
        raise ValueError("No slot columns found. Check `DEFAULT_COLS`.")

    if mode == "all" and all_periods is not None:
        period_sequence = _normalize_period_sequence(all_periods)
    else:
        period_sequence = _normalize_period_sequence(
            df[period_col].dropna().unique().tolist()
        )

    support_rows = []

    for slot in slot_cols:
        # Keep period + one slot column
        temp = df[[period_col, slot]].copy()

        # Convert each cell to list of fillers
        temp[slot] = temp[slot].apply(parse_filler_cell)

        # Explode fillers
        temp = (
            temp
            .explode(slot, ignore_index=True)
            .dropna(subset=[period_col, slot])
            .reset_index(drop=True)
        )

        # Remove empty string fillers
        temp = temp[temp[slot].astype(str).str.strip() != ""]

        if temp.empty:
            if include_zero_slots and mode == "all":
                periods = period_sequence
                for i in range(1, len(periods)):
                    support_rows.append({
                        "slot": slot,
                        "period_1": periods[i - 1],
                        "period_2": periods[i],
                        "support_count": 0.0,
                        "support_weight": 0.0,
                    })
            continue

        if mode == "data_only":
            periods = _normalize_period_sequence(
                temp[period_col].dropna().unique().tolist(),
            )
        else:
            periods = period_sequence

        for i in range(1, len(periods)):
            period_1 = periods[i - 1]
            period_2 = periods[i]

            pair_temp = temp[temp[period_col].isin([period_1, period_2])].copy()
            if pair_temp.empty:
                if include_zero_slots:
                    support_rows.append({
                        "slot": slot,
                        "period_1": period_1,
                        "period_2": period_2,
                        "support_count": 0.0,
                        "support_weight": 0.0,
                    })
                continue

            if min_freq > 1:
                mixed_filler_freq = pair_temp.groupby(slot)[slot].transform("size")
                pair_temp = pair_temp[mixed_filler_freq >= min_freq]

            if pair_temp.empty:
                if include_zero_slots:
                    support_rows.append({
                        "slot": slot,
                        "period_1": period_1,
                        "period_2": period_2,
                        "support_count": 0.0,
                        "support_weight": 0.0,
                    })
                continue

            slot_counts = (
                pair_temp
                .groupby(period_col)
                .size()
                .reindex([period_1, period_2], fill_value=0.0)
                .astype(float)
            )
            c = float(min(
                slot_counts.loc[period_1],
                slot_counts.loc[period_2],
            ))
            w = min(1.0, c / k)

            if not include_zero_slots and c == 0:
                continue

            support_rows.append({
                "slot": slot,
                "period_1": period_1,
                "period_2": period_2,
                "support_count": c,
                "support_weight": w,
            })

    return pd.DataFrame(
        support_rows,
        columns=[
            "slot",
            "period_1",
            "period_2",
            "support_count",
            "support_weight",
        ],
    )
