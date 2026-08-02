"""Utilities for loading HistWords embedding slices."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np


def load_pickle(path: str | Path) -> Any:
    """Load a pickle file using HistWords' latin-1 encoding."""
    with Path(path).open("rb") as file:
        return pickle.load(file, encoding="latin1")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a copy of ``matrix`` with each row scaled to unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HistWordsSlice:
    """One HistWords embedding time slice."""

    def __init__(self, vectors: np.ndarray, vocab: list[str], normalize: bool = True) -> None:
        self.vectors = vectors.astype(np.float32)
        if normalize:
            self.vectors = normalize_rows(self.vectors)
        self.vocab = list(vocab)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

    def has_word(self, word: str) -> bool:
        """Return whether ``word`` is in this slice vocabulary."""
        return word in self.word_to_idx

    def vector(self, word: str) -> np.ndarray:
        """Return the embedding vector for ``word``."""
        if word not in self.word_to_idx:
            raise KeyError(f"OOV word: {word}")
        return self.vectors[self.word_to_idx[word]]


def load_histwords_slice(base_dir: str | Path, year: int) -> HistWordsSlice:
    """Load one HistWords slice from ``{year}-w.npy`` and ``{year}-vocab.pkl``."""
    base_dir = Path(base_dir)
    vectors = np.load(base_dir / f"{year}-w.npy")
    vocab = load_pickle(base_dir / f"{year}-vocab.pkl")
    return HistWordsSlice(vectors, vocab)


def load_histwords_series(base_dir: str | Path, years: list[int]) -> dict[int, HistWordsSlice]:
    """Load HistWords slices for the requested years."""
    return {year: load_histwords_slice(base_dir, year) for year in years}


def load_gensim_slice(
    base_dir: str | Path,
    year: int,
    file_pattern: str | None = None,
    binary: bool | None = None,
    normalize: bool = True,
) -> HistWordsSlice:
    """Load one gensim embedding slice as a ``HistWordsSlice``.

    Args:
        file_pattern: Optional pattern relative to ``base_dir``. It can include
            ``{year}``, for example ``"{year}.model"`` or ``"{year}_vectors.bin"``.
        binary: Whether word2vec-format files are binary. If omitted, ``.bin``
            files are treated as binary and text-like files are treated as text.
    """
    path = _resolve_gensim_path(Path(base_dir), year, file_pattern)
    keyed_vectors = _load_gensim_keyed_vectors(path, binary=binary)
    return HistWordsSlice(
        keyed_vectors.vectors,
        list(keyed_vectors.index_to_key),
        normalize=normalize,
    )


def load_gensim_series(
    base_dir: str | Path,
    years: list[int],
    file_pattern: str | None = None,
    binary: bool | None = None,
    normalize: bool = True,
) -> dict[int, HistWordsSlice]:
    """Load gensim embedding slices for the requested years."""
    return {
        year: load_gensim_slice(
            base_dir=base_dir,
            year=year,
            file_pattern=file_pattern,
            binary=binary,
            normalize=normalize,
        )
        for year in years
    }


def _resolve_gensim_path(base_dir: Path, year: int, file_pattern: str | None) -> Path:
    if file_pattern is not None:
        path = base_dir / file_pattern.format(year=year)
        if not path.exists():
            raise FileNotFoundError(f"Missing gensim embedding file: {path}")
        return path

    candidates = [
        base_dir / f"{year}.model",
        base_dir / f"{year}.kv",
        base_dir / f"{year}_vectors.bin",
        base_dir / f"{year}.bin",
        base_dir / f"{year}.txt",
        base_dir / f"{year}.vec",
    ]
    for path in candidates:
        if path.exists():
            return path

    candidate_list = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No gensim embedding file found. Tried: {candidate_list}")


def _load_gensim_keyed_vectors(path: Path, binary: bool | None):
    from gensim.models import KeyedVectors, Word2Vec

    if path.suffix == ".model":
        try:
            return Word2Vec.load(str(path)).wv
        except Exception:
            return KeyedVectors.load(str(path), mmap="r")

    if path.suffix == ".kv":
        return KeyedVectors.load(str(path), mmap="r")

    is_binary = path.suffix == ".bin" if binary is None else binary
    return KeyedVectors.load_word2vec_format(str(path), binary=is_binary)


def vector_norm(vector: np.ndarray) -> float:
    """Return the L2 norm of one vector as a plain Python float."""
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))
