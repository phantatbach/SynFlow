import pandas as pd
import numpy as np
import ast

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