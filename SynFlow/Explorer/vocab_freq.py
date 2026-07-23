import os
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Optional

import pandas as pd

from SynFlow.const import VALID_FILLER_FORMATS
from SynFlow.utils import format_filler

def freq_suffix(filler_format: str) -> str:
    """Return a filename-safe suffix for a filler format."""
    return filler_format.replace("/", "_")

def count_vocab_file(path: str, filler_format: str) -> Counter:
    """Count vocabulary frequencies according to ``filler_format``."""
    if filler_format not in VALID_FILLER_FORMATS:
        valid_formats = ", ".join(sorted(VALID_FILLER_FORMATS))
        raise ValueError(f"filler_format must be one of: {valid_formats}")

    local_vocab_counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("<s") or line.startswith("</s>"):
                continue
            
            # Split the lines into different parts
            parts = line.split()
            if len(parts) < 6:
                continue
            # Default: wordform, lemma, pos, id, head, deprel
            token, lemma, pos, deprel = parts[0], parts[1], parts[2], parts[5]
            key = format_filler(token, lemma, pos, deprel, filler_format)

            # Count
            local_vocab_counts[key] += 1

    return local_vocab_counts

def count_vocab_parallel_subfolder(
    subfolder_path: str,
    filler_format: str = "lemma/pos",
) -> dict:
    # Get list of file
    all_files = []
    for root, _, files in os.walk(subfolder_path):
        for fname in files:
            all_files.append(os.path.join(root, fname))
    
    # Parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(count_vocab_file, [(path, filler_format) for path in all_files])

    # Combine counters
    total_counter = Counter()
    for counter in results:
        total_counter.update(counter)

    return dict(total_counter)


def vocab_counts_to_df(
    freq_dict: dict,
) -> pd.DataFrame:
    """Convert vocabulary frequency counts for one subfolder to a DataFrame."""
    rows = [
        {
            "vocab": vocab,
            "frequency": frequency,
        }
        for vocab, frequency in sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return pd.DataFrame(rows, columns=["vocab", "frequency"])


def save_freqs(
    freq_df: pd.DataFrame,
    out_folder: str,
    subfolder: str,
    filler_format: str = "lemma/pos",
) -> str:
    """
    Write out `<filler_format>_freq.csv` into out_folder.
    """
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{subfolder}_{freq_suffix(filler_format)}_freq.csv")
    freq_df.to_csv(out_path, index=False)
    return out_path


def gen_vocab_freq(
    corpus_path: str,
    out_folder: str,
    filler_format: str = "lemma/pos",
) -> None:
    """
    Count vocabulary frequencies and save one CSV file for each corpus subfolder.
    """
    out_folder_abs = os.path.abspath(out_folder)
    for subfolder in sorted(os.listdir(corpus_path)):
        subfolder_path = os.path.join(corpus_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        if os.path.abspath(subfolder_path) == out_folder_abs:
            continue
        freqs = count_vocab_parallel_subfolder(subfolder_path, filler_format)
        freq_df = vocab_counts_to_df(freqs)
        save_freqs(freq_df, out_folder, subfolder, filler_format)
