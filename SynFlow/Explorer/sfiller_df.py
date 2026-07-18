from __future__ import annotations

import ast
import os
import re
from ast import literal_eval
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from SynFlow.const import DEFAULT_COLS, DEFAULT_PATTERN, VALID_FILLER_FORMATS
from SynFlow.utils import build_graph, format_filler

# Reformat deprel because build_graph keeps the directions
def reformat_deprel(label: str) -> str:
    """Strip 'chi_' or 'pa_' prefixes from a dependency label."""
    return re.sub(r'^(chi_|pa_)', '', label)

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

def process_file(args) -> List[dict]:
    corpus_folder, fname, pattern, target_lemma, target_pos, slots, filtered_pos, filler_format = args # Use this for multiprocess.Pool
    pattern = pattern or DEFAULT_PATTERN
    
    subfolder = os.path.basename(corpus_folder)  # <— tên subfolder
    filtered_pos = filtered_pos or [] # Guard if filtered_pos is None
    out = []
    path = os.path.join(corpus_folder, fname)

    has_target = False
    has_target_check_string = f'\t{target_lemma}\t{target_pos}'

    with open(path, encoding='utf8') as fh:
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
                            slot_fillers = []
                            # split if there are multiple fillers in a slot
                            for subslot in slot.split("|"):
                                # split your multi-hop slot
                                rel_seq = [r.strip() for r in subslot.split(">")]
                                # get every chain of IDs matching that rel sequence
                                chains  = follow_path(graph, id2deprel, tid, rel_seq)
                                # print(f"DEBUG {fname}:{token_line} chains for {slot} =", chains)

                                # flatten **all** nodes in **all** chains to avoid nested list
                                subslot_fillers = []
                                for chain in chains:
                                    prev_id = tid
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
                                            id2deprel.get((prev_id, nid), 'UNK')
                                            if filler_format.endswith("/deprel")
                                            else None
                                        )
                                        filler = format_filler(token, lemma, pos, deprel, filler_format)

                                        subslot_fillers.append(filler)
                                        prev_id = nid
                                
                                slot_fillers.extend(subslot_fillers)

                            row[slot] = slot_fillers

                        out.append(row)
            else:
                sent_tokens.append(line)
                sent_lines.append(file_line)
                # Check for target lemma/POS in the current line
                if has_target_check_string in line:
                    has_target = True

    return out

# Get all slots from a slot_freq_df to the correct format before building the slot_filler_df
def get_all_slots(df):
    all_slots = "".join(f"[{r}]" for r in df.index)
    return all_slots

def build_sfiller_df(
    corpus_folder: str,
    template: str,
    target_lemma: str,
    target_pos: str,
    filler_format: str = 'lemma/pos',
    num_processes: int = None,
    pattern: re.Pattern = None,
    filtered_pos: list = None,
    output_folder: str = None,
) -> pd.DataFrame:
    """
    1) Walk corpus in parallel, build per-token slot lists.
    2) Drop rows where all slots are empty (write {target}_dropped.txt).
    3) Save the resulting DataFrame to {output_folder}/ and return it.
    """
    pattern   = pattern or DEFAULT_PATTERN
    num_procs = num_processes or max(1, cpu_count()-1)
    slots     = template.strip("[]").split("][")
    filtered_pos = filtered_pos or [] # Guard if filtered_POS is None
    filler_format = filler_format or 'lemma/pos'
    if filler_format not in VALID_FILLER_FORMATS:
        valid_formats = ", ".join(sorted(VALID_FILLER_FORMATS))
        raise ValueError(f"filler_format must be one of: {valid_formats}")
    
    all_rows = []

    # Go through each subfolder in the corpus folder
    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)

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
    df.insert(1, "target", [[target_slot]] * len(df))

    return df, dropped

def replace_in_sfiller_df_column(sfiller_df_path, column_name, replacements, output_path):
    """
    Replace slot-filler values in one CSV column and write the updated CSV.

    The target column is expected to contain string representations of Python
    lists, such as ``"['big/A', 'open/A']"``. Each list item is looked up in
    ``replacements``; matching items are replaced with their mapped value, and
    unmatched items are kept unchanged. Cells that cannot be parsed as lists are
    also left unchanged.

    Args:
        sfiller_df_path (str): Path to the input slot-filler DataFrame.
        column_name (str): Name of the column whose list values should be
            rewritten.
        replacements (dict): Mapping from original filler values to replacement
            values.
        output_path (str): Path where the updated CSV should be saved.
    """
    sfiller_df = pd.read_csv(sfiller_df_path, encoding="utf-8")

    def replace_list_str(cell):
        try:
            items = literal_eval(cell)   # parse '["Open/A", "big/A"]' → list
            if isinstance(items, list):
                return str([replacements.get(x, x) for x in items])
        except Exception:
            pass
        return cell  # nếu parse không được thì giữ nguyên

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
    Merge list-valued slot-filler columns and optionally remove the source columns.

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

    def cell_to_list(cell):
        if isinstance(cell, list):
            return cell
        if isinstance(cell, (tuple, set)):
            return list(cell)
        if pd.isna(cell):
            return []
        if isinstance(cell, str):
            stripped = cell.strip()
            if stripped in ("", "[]"):
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = literal_eval(stripped)
                except (SyntaxError, ValueError):
                    return [cell]
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, (tuple, set)):
                    return list(parsed)
                if parsed is None or pd.isna(parsed):
                    return []
                return [parsed]
            return [cell]
        return [cell]

    def merge_row(row, source_cols):
        merged = []
        seen = set()
        for source_col in source_cols:
            for item in cell_to_list(row[source_col]):
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

#-----------------------------------------------------------
# Extract slot column(s)
def _non_empty(v):
    if isinstance(v, list): return len(v) > 0
    if pd.isna(v): return False
    if isinstance(v, str): return v.strip() not in ("", "[]")
    return True

def extract_slot_cols(spath_df: str, slot_names: list, output_path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(spath_df)
    cols = [c for c in DEFAULT_COLS + slot_names if c in df.columns]
    sub = df[cols].copy()
    keep = sub[slot_names].map(_non_empty).any(axis=1)
    sub = sub[keep]
    if output_path:
        sub.to_csv(output_path, index=False)
    return sub

def explode_slot_col(df: pd.DataFrame, slot_name: str) -> pd.DataFrame:
    """
    Explode a list-valued slot column so each filler occupies one row.
    """
    if slot_name not in df.columns:
        raise KeyError(f"Column not found: {slot_name}")

    exploded_df = df.copy()
    exploded_df[slot_name] = exploded_df[slot_name].apply(_parse_filler_cell)
    exploded_df = exploded_df.explode(slot_name, ignore_index=True)
    exploded_df = exploded_df[exploded_df[slot_name].notna()].reset_index(drop=True)
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

# Compute support of slots across periods
def _count_fillers(cell) -> int:
    """
    Count how many fillers are present in one CSV cell.

    Examples:
        "[]" -> 0
        "['the']" -> 1
        "['white', 'powerful']" -> 2
    """
    if pd.isna(cell):
        return 0

    if isinstance(cell, list):
        return len(cell)

    if isinstance(cell, str):
        cell = cell.strip()

        if cell == "" or cell == "[]":
            return 0

        try:
            parsed = ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            # Fallback: treat a non-empty malformed cell as one filler
            return 1

        if isinstance(parsed, list):
            return len(parsed)

        if parsed is None:
            return 0

        return 1

    return 0

def _sort_period_key(period):
    """
    Sort periods numerically when possible.
    """
    try:
        return int(period)
    except (ValueError, TypeError):
        return str(period)

def _parse_filler_cell(cell):
    """
    Convert one dataframe cell into a list of fillers.

    Handles:
    - "['aboard/A', 'good/A']" -> ['aboard/A', 'good/A']
    - ['aboard/A', 'good/A'] -> ['aboard/A', 'good/A']
    - NaN -> []
    - single string -> [string]
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

    The support count is calculated after applying a period-specific
    minimum filler frequency threshold.

    Example:
        If min_freq = 2 and filler "b" occurs once in period A,
        then "b" is ignored in period A.
        If "b" occurs five times in period B,
        then "b" is kept in period B.

    Steps:
    1. Read a DataFrame where each slot column contains fillers or list-like
       filler cells.
    2. For each slot and period, count individual filler frequencies.
    3. Remove fillers whose frequency is < min_freq within that period.
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
        Minimum frequency of a filler within each period.
        Fillers below this frequency are treated as absent in that period.
        Default: 1.

    mode:
        Period-comparison mode.
        ``"all"`` compares adjacent periods in the complete dataset timeline.
        ``"data_only"`` compares adjacent periods with retained data separately
        for each slot.

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

    # Slot columns are all columns except metadata columns.
    # Also explicitly exclude period_col for safety.
    slot_cols = [
        col for col in df.columns
        if col not in DEFAULT_COLS and col != period_col
    ]

    if not slot_cols:
        raise ValueError("No slot columns found. Check `DEFAULT_COLS`.")

    if mode == "all" and all_periods is not None:
        sorted_periods = sorted(all_periods, key=_sort_period_key)
    else:
        sorted_periods = sorted(
            df[period_col].dropna().unique().tolist(),
            key=_sort_period_key
        )

    # This will contain filtered raw slot counts per period
    period_counts = pd.DataFrame(
        0.0,
        index=sorted_periods,
        columns=slot_cols
    )

    for slot in slot_cols:
        # Keep period + one slot column
        temp = df[[period_col, slot]].copy()

        # Convert each cell to list of fillers
        temp[slot] = temp[slot].apply(_parse_filler_cell)

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
            continue

        # Count each filler within each period
        period_filler_freq = (
            temp
            .groupby([period_col, slot])
            .size()
            .rename("freq")
            .reset_index()
        )

        # Apply period-specific min_freq
        period_filler_freq = period_filler_freq[
            period_filler_freq["freq"] >= min_freq
        ]

        if period_filler_freq.empty:
            continue

        # Aggregate remaining filler counts into slot counts per period
        slot_period_counts = (
            period_filler_freq
            .groupby(period_col)["freq"]
            .sum()
        )

        # Align counts to the selected timeline. In mode="all", this also
        # inserts explicitly requested periods with zero retained data.
        period_counts[slot] = (
            slot_period_counts
            .reindex(sorted_periods, fill_value=0.0)
            .astype(float)
        )

    # Decide which slots to return
    if include_zero_slots:
        output_slots = slot_cols
    else:
        output_slots = [
            slot for slot in slot_cols
            if period_counts[slot].sum() > 0
        ]

    support_rows = []

    for slot in output_slots:
        slot_counts = period_counts[slot]

        if mode == "data_only":
            slot_counts = slot_counts[slot_counts > 0]

        periods = slot_counts.index.tolist()

        for i in range(1, len(periods)):
            period_1 = periods[i - 1]
            period_2 = periods[i]
            c = float(min(
                slot_counts.loc[period_1],
                slot_counts.loc[period_2],
            ))
            w = min(1.0, 2 * c / k)

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
#----------------------------------------------------------------------------------------------------
# # THESE FUNCTIONS HAVE NOT BEEN USED YET

# # Filter the slot fillers by frequency. However, this is the frequency of the whole df and not of individual periods.
# def filter_frequency_sfiller_df(sfiller_df_path, col_name, output_path, min_freq=1):
#     """
#     Filter slot fillers in a DataFrame by their frequency.

#     Parameters:
#         sfiller_df_path (str): Path to the DataFrame CSV file.
#         col_name (str): Name of the column containing the slot fillers.
#         min_freq (int): Minimum frequency of a slot filler to be kept.
#         output_path (str): Path to save the filtered DataFrame.

#     Returns:
#         pd.DataFrame: The filtered DataFrame.

#     Notes:
#         The function overwrites the original file.
#     """
#     df = pd.read_csv(sfiller_df_path)

#     # Convert string representation of list into actual Python list
#     df[col_name] = df[col_name].apply(literal_eval)

#     # Explode into separate rows
#     df = df.explode(col_name).reset_index(drop=True)

#     # Filter by frequency
#     freq = df[col_name].value_counts()
#     df = df[df[col_name].map(freq) >= min_freq]

#     # Sort by subfolder (numeric if possible)
#     df['subfolder'] = pd.to_numeric(df['subfolder'], errors='coerce')
#     df = df.sort_values('subfolder', kind='stable').reset_index(drop=True)

#     # Overwrite file
#     df.to_csv(output_path, index=False)

#     return df

# # Create a pooled slot-filler df based on the pool note
# # Build the year map dict
# def build_year_map(pool_note: dict, slot_name: str) -> dict[int, int]:
#     map = {}
#     for (slot, _block), info in pool_note.items():
#         if slot != slot_name: 
#             continue
#         for group in info["groups"]:
#             tgt = int(group["target"])
#             for src in group["source"]:
#                 map[int(src)] = tgt
#     return map

# # Re map the subfolder column
# def remap_subfolder(df: pd.DataFrame, year_map: dict[int,int]) -> pd.DataFrame:
#     out = df.copy()
#     out["subfolder"] = out["subfolder"].astype(int).map(lambda y: year_map.get(y, y)).astype(str)
#     return out

# def build_pooled_sfiller_df(all_sfillers_csv_path, pool_notes: dict, output_folder) -> pd.DataFrame:
#     file_name = Path(all_sfillers_csv_path).stem

#     df = pd.read_csv(all_sfillers_csv_path)
#     default_cols = DEFAULT_COLS
#     slot_cols = [col for col in df.columns if col not in default_cols]

#     col_dfs = []
#     for col in slot_cols:
#         sub = df[default_cols + [col]].copy()
#         sub = sub[sub[col].notna() & sub[col].astype(str).ne("[]")].reset_index(drop=True)

#         year_map = build_year_map(pool_note=pool_notes, slot_name=col)
#         sub = remap_subfolder(df=sub, year_map=year_map)  # ok kể cả year_map rỗng
#         col_dfs.append(sub)

#     # concat dọc; pandas tự mở rộng cột slot khác nhau
#     pooled_df = pd.concat(col_dfs, ignore_index=True, sort=False)

#     # đưa default_cols lên đầu
#     front = [c for c in default_cols if c in pooled_df.columns]
#     pooled_df = pooled_df[front + [c for c in pooled_df.columns if c not in front]]

#     # replace NaN with empty list
#     for slot_col in slot_cols:
#         pooled_df[slot_col] = pooled_df[slot_col].apply(lambda value: [] if pd.isna(value) else value)

#     # write out
#     output_path = os.path.join(output_folder, file_name + "_pooled.csv")
#     pooled_df.to_csv(output_path, index=False)
#     print(f"Created pooled slot-filler df from {all_sfillers_csv_path} → {output_path}")
#     return pooled_df
