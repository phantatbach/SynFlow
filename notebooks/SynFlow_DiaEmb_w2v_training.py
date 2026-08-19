from pathlib import Path
import sys

import pandas as pd

repo_root = "/home/volt/bach/SynFlow"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from SynFlow.Embedding import train_w2v_folder

training_input_root = Path("/home/volt/bach/Corpora/deu_news_1995-2025_token")
training_output_root = Path("/home/volt/bach/Corpora/deu_news_1995-2025_token_w2v")

w2v_training_results = train_w2v_folder(
    input_root=training_input_root,
    output_root=training_output_root,
    vector_size=300,
    window=4,
    min_count=100,
    max_vocab=50000,
    sg=1,
    negative=5,
    ns_exponent=0.75,
    sample=0,
    seed=42,
    epochs=5,
    process_count=6,
    workers_per_model=1,
    show_progress=True,
    overwrite=False,
)

pd.DataFrame(result.to_dict() for result in w2v_training_results)
