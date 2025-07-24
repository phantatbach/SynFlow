import os
import re
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import List
import json
import ast
from SynFlow.utils import build_graph

DEFAULT_PATTERN = re.compile(
    r'([^\t]+)\t'      # FORM
    r'([^\t]+)\t'      # LEMMA
    r'([^\t])[^\t]*\t' # POS
    r'([^\t]+)\t'      # ID
    r'([^\t]+)\t'      # HEAD
    r'([^\t]+)'        # DEPREL
)

# Reformat deprel because build_graph keeps the directions
def reformat_deprel(label: str) -> str:
    """Strip 'chi_' or 'pa_' prefixes from a dependency label."""
    return re.sub(r'^(chi_|pa_)', '', label)

def process_file(args) -> List[dict]:
    corpus_folder, fname, pattern, target_lemma, target_pos, slots, filtered_pos, filler_format = args # Use this for multiprocess.Pool
    filtered_pos = filtered_pos or [] # Guard if filtered_pos is None
    out = []
    path = os.path.join(corpus_folder, fname)

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

    with open(path, encoding='utf8') as fh:
        file_line = 0
        sent_toks, sent_lines = [], []
        for raw in fh:
            file_line += 1
            line = raw.rstrip("\n")
            if line.startswith("<s id"):
                sent_toks, sent_lines = [], []
            elif line.startswith("</s>"):
                # Build graph when the whole sentence is appended
                id2lemma_pos, graph, id2deprel = build_graph(sent_toks, pattern)
                target_lp = f"{target_lemma}/{target_pos}"
                for tid, lp in id2lemma_pos.items():
                    if lp != target_lp: # Only process the matched token
                        continue
                    token_line = sent_lines[int(tid)-1]
                    row = {"id": f"{target_lemma}/{fname}/{token_line}"}

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

                                    if filler_format == "lemma/deprel":
                                        # ←— CHỖ ĐÃ SỬA —→
                                        raw_label = (
                                            id2deprel.get((prev_id, nid))    # child→parent ⇒ pa_…
                                            or id2deprel.get((nid, prev_id))  # parent→child ⇒ chi_…
                                            or 'UNK'
                                        )
                                        if raw_label.startswith('chi_'):
                                            deprel = reformat_deprel(raw_label)
                                        else:
                                            orig_line = sent_toks[int(nid)-1]
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
                sent_toks.append(line)
                sent_lines.append(file_line)

    return out


def build_slot_df(
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
    out_template_csv: str = "templates.csv"
) -> pd.DataFrame:
    """
    1) Walk corpus in parallel, build per-token slot lists.
    2) Apply frequency filter (freq_path, freq_min, freq_max).
    3) Drop rows where all slots are empty (write {target}_dropped.txt).
    4) Save the resulting DataFrame to out_template_csv and return it.
    """
    pattern   = pattern or DEFAULT_PATTERN
    num_procs = num_processes or max(1, cpu_count()-1)
    slots     = template.strip("[]").split("][")
    fnames    = [f for f in os.listdir(corpus_folder)
                 if f.endswith((".conllu", ".txt"))]
    filtered_pos = filtered_pos or [] # Guard if filtered_POS is None
    filler_format = filler_format or 'lemma/pos'
    args = [
        (corpus_folder, f, pattern, target_lemma, target_pos, slots, filtered_pos, filler_format)
        for f in fnames
    ]
    

    # Parallel file processing
    all_rows = []
    with Pool(num_procs) as pool:
        for rows in pool.imap_unordered(process_file, args, chunksize=10):
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows).set_index("id", drop=True)

    # ensure each slot column exists, even empty columns
    for slot in slots:
        if slot not in df:
            df[slot] = [[]]
    
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
    with open(f"{target_lemma}_dropped.txt","w",encoding='utf8') as f:
        for idx in dropped:
            f.write(idx+"\n")
    df = df[~mask]

    # --- Optional: insert the new "target" slot at column 0 ------------
    target_slot = f"{target_lemma}/{target_pos}"
    # Create a column of single‐item lists [target_slot] for every row:
    df.insert(0, target_slot, [[target_slot]] * len(df))

    # save
    df.to_csv(out_template_csv)
    print(f"Wrote slot‐fillers to {out_template_csv} ({len(df)} rows), "
          f"dropped {len(dropped)} tokens.")
    return df

def sample_slot_df(
    input_csv: str,
    output_csv: str,
    n: int,
    seed: int = 42,
    mode: str = None
) -> pd.DataFrame:
    """
    Read a slot‐filling CSV (with your 'id' as index), sample n rows
    using the given random seed, write them to output_csv, and return
    the sampled DataFrame.
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
                df[col] = df[col].apply(lambda x: eval(x) if pd.notna(x) and isinstance(x, str) else x)
            except Exception as e:
                # Fallback if eval fails (e.g., if it's not a list string)
                print(f"Warning: Could not convert column {col} to list type. Error: {e}")
                # If conversion fails, ensure it's still treated appropriately,
                # e.g., if it's still a string '[]', it will be handled by len(x)==0 check

    target_slot_col_name = df.columns[0]
    slot_cols = [col for col in df.columns if col != target_slot_col_name]

    if mode == 'filled':
        # Filter for rows where ALL identified slot columns are non-empty lists
        # Apply a lambda function row-wise (axis=1)
        # It checks if length of each list in the row's slot-columns is > 0
        # Use boolean masking
        mask_all_filled = df[slot_cols].apply(lambda r: all(len(x) > 0 for x in r), axis=1)
        df = df[mask_all_filled]

    # sample
    sampled = df.sample(n=n, random_state=seed)
    # write out
    sampled.to_csv(output_csv)
    print(f"Sampled {len(sampled)} rows from {input_csv} → {output_csv}")
    return sampled


def build_embeddings(
    df_templates: pd.DataFrame,
    type_embedding_path: str,
    dims: int,
    slot_mode: str,
    tok_mode: str,
    out_embedding: str = "embeddings.csv"
):
    # infer slots as all columns except the index
    slots = list(df_templates.columns)

    # load type embeddings once
    type_df = pd.read_csv(type_embedding_path, index_col=0)
    emb_dict = {lp: type_df.loc[lp].values for lp in type_df.index}

    rows = []
    for fills in df_templates[slots].itertuples(index=False):
        parts = []
        for L in fills:
            # if this cell is a string (e.g. "['we/p']", parse it)
            if isinstance(L, str):
                try:
                    L = ast.literal_eval(L)
                except Exception:
                    L = []
            if L:  # now L is a real list
                vecs = [emb_dict[w] for w in L if w in emb_dict]
                
                # Build Embedding Vectors for Slots
                if slot_mode == 'sum':
                    parts.append(np.sum(vecs, axis=0) if vecs else np.zeros(dims))
                
                elif slot_mode == 'mult':
                    slot_vec = np.ones(dims)
                    for vec in vecs:
                        slot_vec = np.multiply(slot_vec, vec)
                    parts.append(slot_vec)

            else:
                parts.append(np.zeros(dims)) # 0 vectors will be filtered in mult.
        
        # Build Embedding Matrix for Tokens
        if tok_mode == 'concat':
            rows.append(np.concatenate(parts))

        elif tok_mode in ['sum', 'mult']:
            if tok_mode == 'sum':
                rows.append(np.sum(parts, axis=0))
            
            elif tok_mode == 'mult':
                vec = np.ones(dims)
                for part in parts:
                    if not np.all(part == 0):  # Filter out 0 vectors to avoid wiping out the whole product
                        vec = np.multiply(vec, part)
                rows.append(vec)

    emb_arr = np.stack(rows)

    # Build Columns Names
    if tok_mode == 'concat':
        # build column names automatically
        cols = []
        for slot in slots:
            cols += [f"{slot}_{i}" for i in range(dims)]

    elif tok_mode == 'sum' or tok_mode == 'mult':        
        # build column names automatically with the len of the first row (all rows have the same length)
        cols = [f"dim_{i}" for i in range(emb_arr.shape[1])]

    emb_df = pd.DataFrame(emb_arr, columns=cols, index=df_templates.index)
    emb_df.to_csv(f'{out_embedding}_{slot_mode}_{tok_mode}_embedding.csv')
    print(f"Wrote embeddings to {out_embedding}_{slot_mode}_{tok_mode}_embedding.csv (shape {emb_df.shape})")
    return emb_df