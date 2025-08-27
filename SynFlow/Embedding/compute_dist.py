import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_distmtx(emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given emb_df (rows indexed by token ID, columns = embedding dims),
    compute the pairwise cosine‐distance matrix (1 - cosine_similarity).

    Returns a DataFrame dist_df with the same index and columns.
    """
    # ensure numeric matrix
    X = emb_df.values.astype(float)
    # compute cosine‐similarity
    sim = cosine_similarity(X)                # shape (n_tokens, n_tokens)
    sim = np.clip(sim, -1.0, 1.0)

    # convert to distance
    dist = 1.0 - sim
    # build DataFrame
    dist_df = pd.DataFrame(dist,
                           index=emb_df.index,
                           columns=emb_df.index)
    return dist_df