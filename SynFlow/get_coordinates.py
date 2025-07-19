import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, MDS
from umap.umap_ import UMAP

def get_token_coordinates(
    lemma: str,
    dist_df: pd.DataFrame,
    method: str = 'tsne',
    n_components: int = 2,
    random_state: int = 42,
    output_path: str = None,
    n: int = None,
    **kwargs
) -> pd.DataFrame:
    """
    Visualize high-dimensional points from a distance matrix using:
      - 'tsne': t-SNE (sklearn.manifold.TSNE, uses distances as affinities)
      - 'mds': MDS (sklearn.manifold.MDS)
      - 'umap': UMAP (requires umap-learn installed)

    Returns a DataFrame with columns: token_id, x, y.
    """
    # Extract the distance matrix and labels
    D = dist_df.values
    labels = dist_df.index  # these are the token IDs

    method = method.lower()
    if method == 'tsne':
        model = TSNE(
            n_components=n_components,
            metric='precomputed',
            random_state=random_state,
            init="random",
            **kwargs
        )
        coords = model.fit_transform(D)

    elif method == 'mds':
        model = MDS(
            n_components=n_components,
            dissimilarity='precomputed',
            random_state=random_state,
            **kwargs
        )
        coords = model.fit_transform(D)

    elif method == 'umap':
        try:
            import umap.umap_ as umap
        except ImportError:
            raise ImportError("umap-learn is not installed. Install via `pip install umap-learn`.")
        model = umap.UMAP(
            n_components=n_components,
            metric='precomputed',
            random_state=random_state,
            **kwargs
        )
        coords = model.fit_transform(D)

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'tsne', 'mds', or 'umap'.")

    # Build the DataFrame: first create with index=labels, then reset & rename
    coord_df = pd.DataFrame(coords, index=labels, columns=[f"dim{i+1}" for i in range(n_components)])
    coord_df = coord_df.reset_index().rename(columns={
        'index': 'token_id',
        'dim1': 'x',
        'dim2': 'y'
    })
    coord_df.to_csv(f'{output_path}/{lemma}_{n}_{method}.csv', index=False)
    return coord_df
