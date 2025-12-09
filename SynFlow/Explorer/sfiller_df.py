from ast import literal_eval
import os
import re
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List
from SynFlow.utils import build_graph
from .const import DEFAULT_PATTERN

DEFAULT_COLS = ['id', 'subfolder', 'target']
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
        want = rel_seq[i]
        for nb in graph[node]:
            if id2deprel.get((node, nb)) == want: # Check if the edge label is the one we want
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

                                        if filler_format == "lemma/deprel": # This filler is used to get the original deprel of the context words, not the deprel between target and context
                                            raw_label = (
                                                id2deprel.get((prev_id, nid))    # child→parent ⇒ pa_…
                                                or id2deprel.get((nid, prev_id))  # parent→child ⇒ chi_…
                                                or 'UNK'
                                            )
                                            if raw_label.startswith('chi_'):
                                                deprel = reformat_deprel(raw_label)
                                            else:
                                                orig_line = sent_tokens[int(nid)-1] # fall back to the original line
                                                m = pattern.match(orig_line)
                                                raw_field = m.group(6) if m else 'UNK'
                                                deprel = reformat_deprel(raw_field)

                                            filler = f"{lemma}/{deprel}"
                                        else:
                                            filler = f"{lemma}/{pos}"

                                        subslot_fillers.append(filler)
                                        prev_id = nid
                                
                                slot_fillers.extend(subslot_fillers)

                            row[slot] = list(set(slot_fillers))

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
    filler_format: str = 'lemma/pos', # either lemma/pos or lemma/deprel
    num_processes: int = None,
    pattern: re.Pattern = None,
    freq_path: str = None,
    freq_min: int  = 1,
    freq_max: int  = 10**9,
    filtered_pos: list = None,
    output_folder: str = None,
) -> pd.DataFrame:
    """
    1) Walk corpus in parallel, build per-token slot lists.
    2) Apply frequency filter (freq_path, freq_min, freq_max).
    3) Drop rows where all slots are empty (write {target}_dropped.txt).
    4) Save the resulting DataFrame to {output_folder}/ and return it.
    """
    pattern   = pattern or DEFAULT_PATTERN
    num_procs = num_processes or max(1, cpu_count()-1)
    slots     = template.strip("[]").split("][")
    filtered_pos = filtered_pos or [] # Guard if filtered_POS is None
    filler_format = filler_format or 'lemma/pos'
    
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
    
    # Frequency filtering
    if freq_path:
        # 1) load your TSV
        freq = {}
        with open(freq_path, encoding='utf8') as f:
            for line in f:
                lemma_rel, count = line.strip().split('\t')
                freq[lemma_rel] = int(count)

        # 2) filter function using same reformatter
        def keep(w):
            # w is e.g. "one/chi_nsubj" or "one/nsubj"
            lemma, rel = w.split('/',1)
            # normalize the rel before lookup
            rel = reformat_deprel(rel)
            key = f"{lemma}/{rel}"
            # fallback to lemma-only if full key missing
            count = freq.get(key, freq.get(lemma, 0))
            return freq_min <= count <= freq_max

        # 3) apply to each slot
        for slot in slots:
            df[slot] = df[slot].apply(lambda L: [w for w in L if keep(w)])

    # drop empty‐slot rows
    mask = df[slots].apply(lambda r: all(len(x)==0 for x in r), axis=1)
    dropped = df.index[mask].tolist()
    with open(f"{output_folder}/{target_lemma}_dropped.txt","w",encoding='utf8') as f:
        for idx in dropped:
            f.write(idx+"\n")
    df = df[~mask]

    # --- Optional: insert the new "target" slot at column 0 ------------
    target_slot = f"{target_lemma}/{target_pos}"
    # Create a column of single‐item lists [target_slot] for every row:
    df.insert(1, "target", [[target_slot]] * len(df))

    # save
    output_csv = f"{output_folder}/{target_lemma}_samples_sfillerdf_all.csv"
    df.to_csv(output_csv)
    print(f"Wrote slot‐fillers to {output_csv} ({len(df)} rows), "
        f"dropped {len(dropped)} tokens.")
    return df

def sample_sfiller_df(
    input_csv: str,
    output_csv: str,
    n: int,
    seed: int = 42,
    mode: str = None
) -> pd.DataFrame:
    """
    Read a slot‐filling CSV (with your 'id' as index), sample n rows from 
    each subfolder using the given random seed, write them to output_csv,
    and return the sampled DataFrame.
    """
    # load, treating the first column as the index
    df = pd.read_csv(input_csv, index_col=0)

    # Convert string to Python list
    for col in df.columns:
        # Check if the column's values are strings that look like lists
        if df[col].dtype == 'object' and df[col].astype(str).str.startswith('[').any():
            try:
                # Use literal_eval to safely convert string representation of lists
                # or fillna for NaN values which might occur if a slot was truly empty
                df[col] = df[col].apply(lambda x: literal_eval(x) if pd.notna(x) and isinstance(x, str) else x)
            except Exception as e:
                # Fallback if eval fails (e.g., if it's not a list string)
                print(f"Warning: Could not convert column {col} to list type. Error: {e}")
                # If conversion fails, ensure it's still treated appropriately,
                # e.g., if it's still a string '[]', it will be handled by len(x)==0 check
    
    if 'subfolder' not in df.columns:
        raise ValueError("CSV does not contain 'subfolder'.")

    slot_cols = [c for c in df.columns if c not in {'target', 'subfolder', 'id'}]

    if mode == 'filled':
        # Filter for rows where ALL identified slot columns are non-empty lists
        # Apply a lambda function row-wise (axis=1)
        # It checks if length of each list in the row's slot-columns is > 0
        # Use boolean masking
        mask_all_filled = df[slot_cols].apply(lambda r: all(len(x) > 0 for x in r), axis=1)
        df = df[mask_all_filled]
    
    # stratify: tối đa n mỗi subfolder, với seed con ổn định cho từng nhóm
    random_gen = np.random.RandomState(seed)
    per_group_seed = {sf: int(random_gen.randint(0, 2**31-1)) for sf in df['subfolder'].unique()}

    parts = []
    for subf, subf_df in df.groupby('subfolder', group_keys=False):   # ← đúng unpack
        k_rows = min(n, len(subf_df))
        if k_rows > 0:
            parts.append(subf_df.sample(n=k_rows, random_state=per_group_seed[subf]))

    # Collect all sampled rows
    sampled = pd.concat(parts, axis=0) if parts else df.iloc[0:0]
    # write out
    sampled.to_csv(output_csv)
    print(f"Sampled {len(sampled)} rows from {input_csv} → {output_csv}")
    return sampled

def replace_in_sfiller_df_column(sfiller_csv_path, column_name, replacements, output_path=None):
    sfiller_df = pd.read_csv(sfiller_csv_path, encoding="utf-8")

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

def filter_frequency_sfiller(spath_df, col_name, min_freq=1):
    """
    Filter slot fillers in a DataFrame by their frequency.

    Parameters:
        spath_df (str): Path to the DataFrame CSV file.
        col_name (str): Name of the column containing the slot fillers.
        min_freq (int): Minimum frequency of a slot filler to be kept.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Notes:
        The function overwrites the original file.
    """
    df = pd.read_csv(spath_df)

    # Convert string representation of list into actual Python list
    df[col_name] = df[col_name].apply(literal_eval)

    # Explode into separate rows
    df = df.explode(col_name).reset_index(drop=True)

    # Filter by frequency
    freq = df[col_name].value_counts()
    df = df[df[col_name].map(freq) >= min_freq]

    # Sort by subfolder (numeric if possible)
    df['subfolder'] = pd.to_numeric(df['subfolder'], errors='coerce')
    df = df.sort_values('subfolder', kind='stable').reset_index(drop=True)

    # Overwrite file
    df.to_csv(spath_df, index=False)

    return df

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

def extract_1_slot_col(spath_df: str, slot_name: str, output_path: str | None = None) -> pd.DataFrame:
    return extract_slot_cols(spath_df, [slot_name], output_path)
#----------------------------------------------------------------------------------------------------
# Create a pooled slot-filler df based on the pool note
# Build the year map dict
def build_year_map(pool_note: dict, slot_name: str) -> dict[int, int]:
    map = {}
    for (slot, _block), info in pool_note.items():
        if slot != slot_name: 
            continue
        for group in info["groups"]:
            tgt = int(group["target"])
            for src in group["source"]:
                map[int(src)] = tgt
    return map

# Re map the subfolder column
def remap_subfolder(df: pd.DataFrame, year_map: dict[int,int]) -> pd.DataFrame:
    out = df.copy()
    out["subfolder"] = out["subfolder"].astype(int).map(lambda y: year_map.get(y, y)).astype(str)
    return out


# THIS FUNCTION HAS NOT BEEN USED YET
def build_pooled_sfiller_df(all_sfillers_csv_path, pool_notes: dict, output_folder) -> pd.DataFrame:
    file_name = Path(all_sfillers_csv_path).stem

    df = pd.read_csv(all_sfillers_csv_path)
    default_cols = DEFAULT_COLS
    slot_cols = [col for col in df.columns if col not in default_cols]

    col_dfs = []
    for col in slot_cols:
        sub = df[default_cols + [col]].copy()
        sub = sub[sub[col].notna() & sub[col].astype(str).ne("[]")].reset_index(drop=True)

        year_map = build_year_map(pool_note=pool_notes, slot_name=col)
        sub = remap_subfolder(df=sub, year_map=year_map)  # ok kể cả year_map rỗng
        col_dfs.append(sub)

    # concat dọc; pandas tự mở rộng cột slot khác nhau
    pooled_df = pd.concat(col_dfs, ignore_index=True, sort=False)

    # đưa default_cols lên đầu
    front = [c for c in default_cols if c in pooled_df.columns]
    pooled_df = pooled_df[front + [c for c in pooled_df.columns if c not in front]]

    # replace NaN with empty list
    for slot_col in slot_cols:
        pooled_df[slot_col] = pooled_df[slot_col].apply(lambda value: [] if pd.isna(value) else value)

    # write out
    output_path = os.path.join(output_folder, file_name + "_pooled.csv")
    pooled_df.to_csv(output_path, index=False)
    print(f"Created pooled slot-filler df from {all_sfillers_csv_path} → {output_path}")
    return pooled_df
