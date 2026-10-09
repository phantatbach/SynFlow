"""Core Word2Vec training utilities for period-split sentence folders."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Literal
import multiprocessing
import random
import tempfile
import warnings

from gensim import utils
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

DEFAULT_SAVE_FORMATS = ("model", "keyed_vectors", "vectors_bin", "vectors_txt")
RAW_VECTOR_SAVE_FORMATS = ("keyed_vectors", "vectors_bin", "vectors_txt")
#----------------------------------------------
# Word2Vec training for period-split sentence folders
#----------------------------------------------

@dataclass(frozen=True)
class W2VTrainingResult:
    """Training output metadata for one source text file."""

    input_path: Path
    output_path: Path
    keyed_vectors_path: Path
    vectors_bin_path: Path
    vectors_txt_path: Path
    sentence_count: int
    vocabulary_size: int

    def to_dict(self) -> dict[str, str | int]:
        """Return the result as plain values for notebook display."""
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "keyed_vectors_path": str(self.keyed_vectors_path),
            "vectors_bin_path": str(self.vectors_bin_path),
            "vectors_txt_path": str(self.vectors_txt_path),
            "sentence_count": self.sentence_count,
            "vocabulary_size": self.vocabulary_size,
        }


@dataclass(frozen=True)
class _W2VTrainingJob:
    input_path: Path
    output_path: Path
    keyed_vectors_path: Path
    vectors_bin_path: Path
    vectors_txt_path: Path
    vector_size: int
    window: int
    min_count: int
    max_vocab: int | None
    sg: int
    negative: int
    ns_exponent: float
    sample: float
    seed: int
    epochs: int
    workers: int
    save_formats: tuple[str, ...]
    lowercase: bool
    overwrite: bool


class WhitespaceSentenceIterator:
    """Stream whitespace-tokenized sentences from a plain text file."""

    def __init__(self, path: str | Path, lowercase: bool = False) -> None:
        self.path = Path(path)
        self.lowercase = lowercase

    def __iter__(self) -> Iterable[list[str]]:
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                if self.lowercase:
                    text = text.lower()
                tokens = text.split()
                if tokens:
                    yield tokens


def train_w2v_folder(
    input_root: str | Path,
    output_root: str | Path,
    *,  # Force the following arguments to be keyword-only for clarity.
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    max_vocab: int | None = None,
    sg: int = 1,
    negative: int = 5,
    ns_exponent: float = 0.75,
    sample: float = 1e-5,
    seed: int = 1,
    epochs: int = 5,
    process_count: int | None = None,
    workers_per_model: int = 1,
    model_filename: str = "{name}.model",
    save_formats: tuple[str, ...] = DEFAULT_SAVE_FORMATS,
    show_progress: bool = True,
    lowercase: bool = False,
    overwrite: bool = False,
) -> list[W2VTrainingResult]:
    """Train one Word2Vec model for each period subfolder below ``input_root``.

    The input root must contain period/category subfolders with exactly one
    ``.txt`` file each. Text files directly inside ``input_root`` are ignored.
    Each line in a text file is treated as one whitespace-tokenized sentence.
    Models are saved below ``output_root`` using the same relative subfolder
    layout as the input.

    Args:
        input_root: Top-level input folder. The function reads direct
            subfolders such as ``input_root/1900/`` and ``input_root/1910/``.
            Each subfolder must contain exactly one ``.txt`` file.
        output_root: Top-level output folder. Each model is written under the
            same relative subfolder as its input text file, for example
            ``output_root/1900/1900.model``.
        vector_size: Embedding dimensionality passed to gensim ``Word2Vec``.
        window: Maximum context-window distance around each target word.
        min_count: Minimum token frequency required for a word to enter the
            model vocabulary.
        max_vocab: Maximum final vocabulary size for each period model. If set,
            the trainer keeps the most frequent tokens up to this limit after
            applying ``min_count``. For example, ``max_vocab=50000`` keeps at
            most the top 50,000 tokens per period.
        sg: Training algorithm flag passed to gensim. Use ``1`` for skip-gram
            and ``0`` for CBOW.
        negative: Number of negative samples used by negative sampling. Set to
            ``0`` to disable negative sampling.
        ns_exponent: Exponent used to shape the negative-sampling distribution.
            ``0.75`` is the common SGNS setting used by Mikolov et al. and
            Hamilton et al.
        sample: Threshold for random downsampling of frequent words. Set to
            ``0`` to disable subsampling.
        seed: Random seed used for sentence shuffling and passed to gensim for
            reproducible initialization and sampling.
        epochs: Number of training passes over each text file.
        process_count: Number of text files to train in parallel. Defaults to
            the smaller of CPU count and number of discovered text files.
        workers_per_model: Gensim worker threads inside each training process.
            Keep this low when using many processes to avoid CPU oversubscription.
        model_filename: Filename template for each saved model. Supported
            fields are ``{name}`` for the input subfolder name and ``{stem}``
            for the source text filename without ``.txt``.
        save_formats: Output formats to write for each trained model. Defaults
            to all supported formats: ``"model"`` for the full gensim
            ``Word2Vec`` model, ``"keyed_vectors"`` for gensim ``.kv`` vectors,
            ``"vectors_bin"`` for word2vec binary format, and ``"vectors_txt"``
            for word2vec text format.
        show_progress: Whether to show a tqdm progress bar over completed
            period subfolders. If tqdm is unavailable, training continues
            without a progress bar.
        lowercase: Whether to lowercase sentence text before whitespace
            tokenization. Defaults to preserving the original casing.
        overwrite: Whether to retrain and replace an existing output model.
            When ``False``, existing models are loaded and reported instead of
            retrained.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    text_paths = _discover_training_files(input_root)

    if not text_paths:
        raise FileNotFoundError(f"No .txt training files found below: {input_root}")

    save_formats = _validate_save_formats(save_formats)
    _validate_max_vocab(max_vocab)
    _validate_word2vec_parameters(
        negative=negative,
        ns_exponent=ns_exponent,
        sample=sample,
    )
    process_total = _resolve_process_count(process_count, len(text_paths))
    jobs = [
        _build_training_job(
            input_root=input_root,
            output_root=output_root,
            text_path=text_path,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            max_vocab=max_vocab,
            sg=sg,
            negative=negative,
            ns_exponent=ns_exponent,
            sample=sample,
            seed=seed,
            epochs=epochs,
            workers=workers_per_model,
            model_filename=model_filename,
            save_formats=save_formats,
            lowercase=lowercase,
            overwrite=overwrite,
        )
        for text_path in text_paths
    ]

    if process_total == 1:
        iterator = _progress_iter(jobs, total=len(jobs), enabled=show_progress)
        return [_train_one_model(job) for job in iterator]

    results: list[W2VTrainingResult] = []
    with ProcessPoolExecutor(max_workers=process_total) as executor:
        future_to_job = {executor.submit(_train_one_model, job): job for job in jobs}
        futures = as_completed(future_to_job)
        iterator = _progress_iter(futures, total=len(future_to_job), enabled=show_progress)
        for future in iterator:
            results.append(future.result())

    return sorted(results, key=lambda result: result.input_path)


def _discover_training_files(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_root}")

    text_paths = []
    subfolders = sorted(path for path in input_root.iterdir() if path.is_dir())
    for subfolder in subfolders:
        subfolder_texts = sorted(path for path in subfolder.glob("*.txt") if path.is_file())
        if not subfolder_texts:
            raise FileNotFoundError(f"No .txt training file found in subfolder: {subfolder}")
        if len(subfolder_texts) > 1:
            paths = ", ".join(str(path) for path in subfolder_texts)
            raise ValueError(
                f"Expected exactly one .txt training file in {subfolder}; found: {paths}"
            )
        text_paths.append(subfolder_texts[0])

    return text_paths


def _resolve_process_count(process_count: int | None, job_count: int) -> int:
    if process_count is not None:
        if process_count < 1:
            raise ValueError("process_count must be at least 1.")
        return min(process_count, job_count)

    cpu_count = multiprocessing.cpu_count()
    return max(1, min(cpu_count, job_count))


def _validate_save_formats(save_formats: tuple[str, ...]) -> tuple[str, ...]:
    valid_formats = set(DEFAULT_SAVE_FORMATS)
    unknown_formats = sorted(set(save_formats) - valid_formats)
    if unknown_formats:
        joined = ", ".join(unknown_formats)
        raise ValueError(f"Unknown save format(s): {joined}")
    return tuple(dict.fromkeys(save_formats))


def _validate_raw_vector_save_formats(save_formats: tuple[str, ...]) -> tuple[str, ...]:
    valid_formats = set(RAW_VECTOR_SAVE_FORMATS)
    requested_formats = tuple(format_name for format_name in save_formats if format_name != "model")
    unknown_formats = sorted(set(requested_formats) - valid_formats)
    if unknown_formats:
        joined = ", ".join(unknown_formats)
        raise ValueError(f"Unknown raw vector save format(s): {joined}")

    deduplicated = tuple(dict.fromkeys(requested_formats))
    if not deduplicated:
        raise ValueError("save_formats must contain at least one raw vector format.")
    return deduplicated


def _validate_max_vocab(max_vocab: int | None) -> None:
    if max_vocab is not None and max_vocab < 1:
        raise ValueError("max_vocab must be at least 1 when provided.")


def _validate_word2vec_parameters(
    *,
    negative: int,
    ns_exponent: float,
    sample: float,
) -> None:
    if negative < 0:
        raise ValueError("negative must be greater than or equal to 0.")
    if ns_exponent < 0:
        raise ValueError("ns_exponent must be greater than or equal to 0.")
    if sample < 0:
        raise ValueError("sample must be greater than or equal to 0.")


def _progress_iter(iterable, *, total: int, enabled: bool):
    if not enabled:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, total=total, desc="Progress", unit="period")


def _build_training_job(
    *,
    input_root: Path,
    output_root: Path,
    text_path: Path,
    vector_size: int,
    window: int,
    min_count: int,
    max_vocab: int | None,
    sg: int,
    negative: int,
    ns_exponent: float,
    sample: float,
    seed: int,
    epochs: int,
    workers: int,
    model_filename: str,
    save_formats: tuple[str, ...],
    lowercase: bool,
    overwrite: bool,
) -> _W2VTrainingJob:
    relative_parent = text_path.parent.relative_to(input_root)
    output_dir = output_root / relative_parent
    filename = model_filename.format(name=text_path.parent.name, stem=text_path.stem)
    output_path = output_dir / filename
    output_stem = output_path.with_suffix("")
    return _W2VTrainingJob(
        input_path=text_path,
        output_path=output_path,
        keyed_vectors_path=output_stem.with_suffix(".kv"),
        vectors_bin_path=output_stem.with_name(f"{output_stem.name}_vectors.bin"),
        vectors_txt_path=output_stem.with_name(f"{output_stem.name}_vectors.txt"),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        max_vocab=max_vocab,
        sg=sg,
        negative=negative,
        ns_exponent=ns_exponent,
        sample=sample,
        seed=seed,
        epochs=epochs,
        workers=workers,
        save_formats=save_formats,
        lowercase=lowercase,
        overwrite=overwrite,
    )


def _train_one_model(job: _W2VTrainingJob) -> W2VTrainingResult:
    if job.output_path.exists() and not job.overwrite:
        model = Word2Vec.load(str(job.output_path))
        _save_model_outputs(model, job)
        return W2VTrainingResult(
            input_path=job.input_path,
            output_path=job.output_path,
            keyed_vectors_path=job.keyed_vectors_path,
            vectors_bin_path=job.vectors_bin_path,
            vectors_txt_path=job.vectors_txt_path,
            sentence_count=_count_sentences(job.input_path),
            vocabulary_size=len(model.wv),
        )

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    top_vocab = _collect_top_vocab(
        job.input_path,
        lowercase=job.lowercase,
        min_count=job.min_count,
        max_vocab=job.max_vocab,
    )
    temp_path = _write_shuffled_training_file(job)
    try:
        model = Word2Vec(
            sentences=LineSentence(str(temp_path)),
            vector_size=job.vector_size,
            window=job.window,
            min_count=job.min_count,
            trim_rule=_build_trim_rule(top_vocab),
            sg=job.sg,
            negative=job.negative,
            ns_exponent=job.ns_exponent,
            sample=job.sample,
            seed=job.seed,
            workers=job.workers,
            epochs=job.epochs,
        )
        _save_model_outputs(model, job)
    finally:
        temp_path.unlink(missing_ok=True)

    return W2VTrainingResult(
        input_path=job.input_path,
        output_path=job.output_path,
        keyed_vectors_path=job.keyed_vectors_path,
        vectors_bin_path=job.vectors_bin_path,
        vectors_txt_path=job.vectors_txt_path,
        sentence_count=model.corpus_count,
        vocabulary_size=len(model.wv),
    )


def _write_shuffled_training_file(job: _W2VTrainingJob) -> Path:
    lines = _read_training_lines(job.input_path, lowercase=job.lowercase)
    random.Random(job.seed).shuffle(lines)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=job.output_path.parent,
        prefix=f".{job.output_path.stem}.shuffled.",
        suffix=".txt",
    ) as file:
        file.writelines(f"{line}\n" for line in lines)
        return Path(file.name)


def _read_training_lines(path: Path, *, lowercase: bool) -> list[str]:
    lines = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            if lowercase:
                text = text.lower()
            lines.append(text)
    return lines


def _save_model_outputs(model: Word2Vec, job: _W2VTrainingJob) -> None:
    if "model" in job.save_formats and (job.overwrite or not job.output_path.exists()):
        model.save(str(job.output_path))
    if "keyed_vectors" in job.save_formats and (
        job.overwrite or not job.keyed_vectors_path.exists()
    ):
        model.wv.save(str(job.keyed_vectors_path))
    if "vectors_bin" in job.save_formats and (
        job.overwrite or not job.vectors_bin_path.exists()
    ):
        model.wv.save_word2vec_format(str(job.vectors_bin_path), binary=True)
    if "vectors_txt" in job.save_formats and (
        job.overwrite or not job.vectors_txt_path.exists()
    ):
        model.wv.save_word2vec_format(str(job.vectors_txt_path), binary=False)


def _collect_top_vocab(
    path: Path,
    *,
    lowercase: bool,
    min_count: int,
    max_vocab: int | None,
) -> set[str] | None:
    if max_vocab is None:
        return None

    counts: Counter[str] = Counter()
    for sentence in WhitespaceSentenceIterator(path, lowercase=lowercase):
        counts.update(sentence)

    ranked_tokens = sorted(
        (item for item in counts.items() if item[1] >= min_count),
        key=lambda item: (-item[1], item[0]),
    )
    return {token for token, _ in ranked_tokens[:max_vocab]}


def _build_trim_rule(top_vocab: set[str] | None):
    if top_vocab is None:
        return None

    def trim_rule(word: str, count: int, min_count: int) -> int:
        if count < min_count:
            return utils.RULE_DISCARD
        if word in top_vocab:
            return utils.RULE_KEEP
        return utils.RULE_DISCARD

    return trim_rule


def _count_sentences(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


#----------------------------------------------
# Hamilton-style sequential orthogonal Procrustes alignment
#----------------------------------------------

@dataclass(frozen=True)
class W2VAlignmentResult:
    """Alignment output metadata for one period model."""

    period: str | int
    input_path: Path
    output_path: Path
    keyed_vectors_path: Path
    vectors_bin_path: Path
    vectors_txt_path: Path
    aligned_to_period: str | int | None
    anchor_count: int
    vocabulary_size: int

    def to_dict(self) -> dict[str, str | int | None]:
        """Return the result as plain values for notebook display."""
        return {
            "period": self.period,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "keyed_vectors_path": str(self.keyed_vectors_path),
            "vectors_bin_path": str(self.vectors_bin_path),
            "vectors_txt_path": str(self.vectors_txt_path),
            "aligned_to_period": self.aligned_to_period,
            "anchor_count": self.anchor_count,
            "vocabulary_size": self.vocabulary_size,
        }


def align_w2v_folder(
    input_root: str | Path,
    output_root: str | Path,
    periods: list[str | int] | None = None,
    *,
    model_filename: str = "{period}.model",
    save_formats: tuple[str, ...] = DEFAULT_SAVE_FORMATS,
    min_anchor_count: int | None = None,
    top_k_anchor: int | None = None,
    overwrite: bool = False,
) -> list[W2VAlignmentResult]:
    """Sequentially align period Word2Vec models using orthogonal Procrustes.

    The input root must contain one subfolder per period, and each period
    subfolder must contain the model specified by ``model_filename``. Vectors
    are L2-normalized as full matrices before alignment, matching the HistWords
    default loading behavior. The first period is normalized and saved as the
    base space. Each later period is aligned to the previously aligned period.

    Args:
        input_root: Root folder containing period subfolders with trained W2V
            models.
        output_root: Root folder where aligned period subfolders are written.
        periods: Ordered periods to align. If omitted, the order is inferred
            from direct subfolder names under ``input_root`` sorted by name.
        model_filename: Filename template inside each period subfolder. It can
            include ``{period}``, for example ``"{period}.model"``.
        save_formats: Output formats to write for each aligned model. Defaults
            to all supported formats.
        min_anchor_count: Minimum number of shared words selected as anchors.
            If fewer shared words are available, all of them are used. If
            ``None``, the embedding dimensionality is used as the minimum.
        top_k_anchor: Target number of shared words used as anchors. Words are
            ranked by the smaller of their frequencies in the two periods. A
            larger ``min_anchor_count`` takes precedence. If ``None``, all
            shared words are used.
        overwrite: Whether to replace existing aligned output models. Set to
            ``True`` when changing ``top_k_anchor`` for outputs already saved.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    if periods is None:
        periods = _discover_period_subfolders(input_root)
    if not periods:
        raise ValueError("No periods provided or discovered for W2V alignment.")

    save_formats = _validate_save_formats(save_formats)
    if min_anchor_count is not None and min_anchor_count < 1:
        raise ValueError("min_anchor_count must be at least 1.")
    if top_k_anchor is not None and top_k_anchor < 1:
        raise ValueError("top_k_anchor must be at least 1.")

    results: list[W2VAlignmentResult] = []
    previous_model: Word2Vec | None = None
    previous_period: str | int | None = None

    iterator = _progress_iter(periods, total=len(periods), enabled=True)
    for period in iterator:
        input_path = _resolve_period_model_path(input_root, period, model_filename)
        paths = _build_alignment_output_paths(output_root, period)

        if paths["model"].exists() and not overwrite:
            current_model = Word2Vec.load(str(paths["model"]))
            _normalize_keyed_vectors(current_model.wv)
            _save_aligned_model_outputs(
                model=current_model,
                paths=paths,
                save_formats=save_formats,
                overwrite=False,
            )
            anchor_count = 0
            if previous_model is not None:
                anchor_count = _count_common_vocab(previous_model, current_model)
                _warn_if_below_minimum_anchor_count(
                    available_count=anchor_count,
                    vector_size=previous_model.wv.vector_size,
                    min_anchor_count=min_anchor_count,
                    scope=f"Embedding space {previous_period!r} -> {period!r}",
                )
                anchor_count = _selected_anchor_count(
                    shared_count=anchor_count,
                    vector_size=previous_model.wv.vector_size,
                    top_k_anchor=top_k_anchor,
                    min_anchor_count=min_anchor_count,
                )
            results.append(
                _build_alignment_result(
                    period=period,
                    input_path=input_path,
                    paths=paths,
                    aligned_to_period=previous_period,
                    anchor_count=anchor_count,
                    vocabulary_size=len(current_model.wv),
                )
            )
            previous_model = current_model
            previous_period = period
            continue

        current_model = Word2Vec.load(str(input_path))
        _normalize_keyed_vectors(current_model.wv)

        anchor_count = 0
        if previous_model is not None:
            rotation, anchor_count = _orthogonal_procrustes_rotation(
                base_model=previous_model,
                other_model=current_model,
                min_anchor_count=min_anchor_count,
                top_k_anchor=top_k_anchor,
                warning_scope=f"Embedding space {previous_period!r} -> {period!r}",
            )
            current_model.wv.vectors = current_model.wv.vectors.dot(rotation).astype(
                np.float32,
                copy=False,
            )
            _reset_keyed_vector_norms(current_model.wv)

        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        _save_aligned_model_outputs(
            model=current_model,
            paths=paths,
            save_formats=save_formats,
            overwrite=overwrite,
        )
        results.append(
            _build_alignment_result(
                period=period,
                input_path=input_path,
                paths=paths,
                aligned_to_period=previous_period,
                anchor_count=anchor_count,
                vocabulary_size=len(current_model.wv),
            )
        )
        previous_model = current_model
        previous_period = period

    return results


def align_w2v_raw_vec_folder(
    input_root: str | Path,
    output_root: str | Path,
    periods: list[str | int] | None = None,
    *,
    vector_filename: str = "{period}.txt",
    vocab_freq_filename: str | None = None,
    save_formats: tuple[str, ...] = RAW_VECTOR_SAVE_FORMATS,
    min_anchor_count: int | None = None,
    top_k_anchor: int | None = None,
    overwrite: bool = False,
) -> list[W2VAlignmentResult]:
    """Sequentially align raw word2vec text vectors using orthogonal Procrustes.

    This follows the same period-folder, normalization, skipping, and output
    path contract as :func:`align_w2v_folder`, but each input period file is
    loaded with ``KeyedVectors.load_word2vec_format(..., binary=False)``. When
    ``top_k_anchor`` is set, word counts come from separate vocabulary files.
    Because raw vector text files do not contain full Word2Vec training state,
    this function does not write a ``.model`` output.

    Args:
        input_root: Root folder containing period subfolders with raw word2vec
            text vector files.
        output_root: Root folder where aligned period subfolders are written.
        periods: Ordered periods to align. If omitted, the order is inferred
            from direct subfolder names under ``input_root`` sorted by name.
        vector_filename: Filename template inside each period subfolder. It can
            include ``{period}``, for example ``"{period}_vectors.txt"``.
        vocab_freq_filename: Path relative to each period subfolder for a
            two-column tab-separated ``word<tab>count`` file, or a CSV with
            ``vocab,frequency`` columns. Required when ``top_k_anchor`` is set.
            The path may include ``{period}``.
        save_formats: Output formats to write for each aligned vector file.
            Defaults to all supported raw-vector formats: ``"keyed_vectors"``,
            ``"vectors_bin"``, and ``"vectors_txt"``.
        min_anchor_count: Minimum number of shared words selected as anchors.
            If fewer shared words are available, all of them are used. If
            ``None``, the embedding dimensionality is used as the minimum.
        top_k_anchor: Target number of shared words used as anchors, ranked by
            the smaller frequency in the two periods. A larger
            ``min_anchor_count`` takes precedence. If ``None``, all shared
            words are used.
        overwrite: Whether to replace existing aligned output files. Set to
            ``True`` when changing ``top_k_anchor`` for outputs already saved.
    """
    return _align_raw_vec_folder(
        input_root=Path(input_root),
        output_root=Path(output_root),
        periods=periods,
        vector_filename=vector_filename,
        vocab_freq_filename=vocab_freq_filename,
        save_formats=save_formats,
        min_anchor_count=min_anchor_count,
        top_k_anchor=top_k_anchor,
        top_k_anchor_pct=None,
        dependency_alignment_mode=None,
        overwrite=overwrite,
    )


def align_depw2v_raw_vec_folder(
    input_root: str | Path,
    output_root: str | Path,
    periods: list[str | int] | None = None,
    *,
    vector_filename: str = "{period}.txt",
    vocab_freq_filename: str,
    top_k_anchor: int = 5000,
    top_k_anchor_pct: float | None = None,
    alignment_mode: Literal["global", "region_specific"] = "global",
    save_formats: tuple[str, ...] = RAW_VECTOR_SAVE_FORMATS,
    min_anchor_count: int | None = None,
    overwrite: bool = False,
) -> list[W2VAlignmentResult]:
    """Align dependency vectors globally or separately by relation.

    In ``"global"`` mode, relation quotas are proportional to the smaller of
    their total frequencies in the two periods. Shared ``item/relation`` keys
    are ranked within each relation by their smaller cross-period frequency,
    and all selected anchors estimate one global rotation. Unfilled quotas are
    not redistributed unless more anchors are needed to satisfy
    ``min_anchor_count``.

    In ``"region_specific"`` mode, each relation uses the top
    ``top_k_anchor_pct`` percent of its own ranked shared items to estimate a
    separate rotation. That rotation is applied to every current-period vector
    in the relation.

    Vectors from different relations no longer share one aligned coordinate
    system and should only be compared within their own relation.

    Args:
        input_root: Root folder containing one raw-vector subfolder per period.
        output_root: Root folder where aligned period subfolders are written.
        periods: Ordered periods to align. Folder names are used when omitted.
        vector_filename: Raw word2vec text filename template for each period.
        vocab_freq_filename: Path relative to each period subfolder for a
            two-column tab-separated ``item<tab>count`` file, or a CSV with
            ``vocab,frequency`` columns. The path may include ``{period}``.
        top_k_anchor: Target total anchor count before relation-level rounding
            in global mode. It is ignored in region-specific mode.
        top_k_anchor_pct: Percentage of shared items selected independently
            within each relation in region-specific mode. For example, ``5``
            selects the top 5 percent. It must be omitted in global mode.
        alignment_mode: Whether to estimate one global rotation or one rotation
            for each dependency relation.
        save_formats: Raw-vector output formats to write.
        min_anchor_count: Minimum selected anchor count overall in global mode
            and within each relation in region-specific mode. If fewer shared
            items are available, all available items are used. The embedding
            dimensionality is used as the minimum when omitted.
        overwrite: Whether to replace existing aligned output files. Set to
            ``True`` after changing anchor settings for existing outputs.
    """
    if alignment_mode not in {"global", "region_specific"}:
        raise ValueError("alignment_mode must be 'global' or 'region_specific'.")

    return _align_raw_vec_folder(
        input_root=Path(input_root),
        output_root=Path(output_root),
        periods=periods,
        vector_filename=vector_filename,
        vocab_freq_filename=vocab_freq_filename,
        save_formats=save_formats,
        min_anchor_count=min_anchor_count,
        top_k_anchor=top_k_anchor,
        top_k_anchor_pct=top_k_anchor_pct,
        dependency_alignment_mode=alignment_mode,
        overwrite=overwrite,
    )


def _align_raw_vec_folder(
    *,
    input_root: Path,
    output_root: Path,
    periods: list[str | int] | None,
    vector_filename: str,
    vocab_freq_filename: str | None,
    save_formats: tuple[str, ...],
    min_anchor_count: int | None,
    top_k_anchor: int | None,
    top_k_anchor_pct: float | None,
    dependency_alignment_mode: Literal["global", "region_specific"] | None,
    overwrite: bool,
) -> list[W2VAlignmentResult]:
    if periods is None:
        periods = _discover_period_subfolders(input_root)
    if not periods:
        raise ValueError("No periods provided or discovered for W2V alignment.")

    save_formats = _validate_raw_vector_save_formats(save_formats)
    if min_anchor_count is not None and min_anchor_count < 1:
        raise ValueError("min_anchor_count must be at least 1.")
    if dependency_alignment_mode not in {None, "global", "region_specific"}:
        raise ValueError("alignment_mode must be 'global' or 'region_specific'.")
    if dependency_alignment_mode == "global":
        if top_k_anchor is None or top_k_anchor < 1:
            raise ValueError("top_k_anchor must be at least 1 in global mode.")
        if top_k_anchor_pct is not None:
            raise ValueError("top_k_anchor_pct must be omitted in global mode.")
    elif dependency_alignment_mode == "region_specific":
        if top_k_anchor_pct is None or not 0 < top_k_anchor_pct <= 100:
            raise ValueError(
                "top_k_anchor_pct must be greater than 0 and at most 100 "
                "in region-specific mode."
            )
    else:
        if top_k_anchor is not None and top_k_anchor < 1:
            raise ValueError("top_k_anchor must be at least 1.")
        if top_k_anchor_pct is not None:
            raise ValueError("top_k_anchor_pct is only supported for dependency alignment.")

    needs_frequencies = (
        top_k_anchor is not None or dependency_alignment_mode is not None
    )
    if needs_frequencies and not vocab_freq_filename:
        raise ValueError("vocab_freq_filename is required for frequency-based anchors.")

    results: list[W2VAlignmentResult] = []
    previous_vectors: KeyedVectors | None = None
    previous_frequencies: dict[str, int] | None = None
    previous_period: str | int | None = None

    iterator = _progress_iter(periods, total=len(periods), enabled=True)
    for period in iterator:
        input_path = _resolve_period_vector_path(input_root, period, vector_filename)
        paths = _build_alignment_output_paths(output_root, period)
        current_frequencies = None
        if needs_frequencies:
            frequency_path = _resolve_period_vocab_frequency_path(
                input_root, period, vocab_freq_filename
            )
            current_frequencies = _load_vocab_frequencies(frequency_path)

        existing_path = _find_existing_raw_vector_output(paths, save_formats)
        if existing_path is not None and not overwrite:
            current_vectors = _load_existing_raw_vector_output(existing_path, paths)
            _normalize_keyed_vectors(current_vectors)
            _save_aligned_keyed_vector_outputs(
                keyed_vectors=current_vectors,
                paths=paths,
                save_formats=save_formats,
                overwrite=False,
            )
            anchor_count = 0
            if previous_vectors is not None:
                if dependency_alignment_mode == "global":
                    anchors = _select_dependency_anchors(
                        base_vectors=previous_vectors,
                        other_vectors=current_vectors,
                        base_frequencies=previous_frequencies,
                        other_frequencies=current_frequencies,
                        top_k_anchor=top_k_anchor,
                        min_anchor_count=min_anchor_count,
                        warning_scope=(
                            f"Dependency embedding space "
                            f"{previous_period!r} -> {period!r}"
                        ),
                    )
                    anchor_count = len(anchors)
                elif dependency_alignment_mode == "region_specific":
                    anchors_by_relation = _select_dependency_anchors_by_relation(
                        base_vectors=previous_vectors,
                        other_vectors=current_vectors,
                        base_frequencies=previous_frequencies,
                        other_frequencies=current_frequencies,
                        top_k_anchor_pct=top_k_anchor_pct,
                        min_anchor_count=min_anchor_count,
                        warning_scope=(
                            f"Dependency regions {previous_period!r} -> {period!r}"
                        ),
                    )
                    anchor_count = sum(map(len, anchors_by_relation.values()))
                else:
                    anchor_count = _count_common_keyed_vectors(
                        previous_vectors, current_vectors
                    )
                    _warn_if_below_minimum_anchor_count(
                        available_count=anchor_count,
                        vector_size=previous_vectors.vector_size,
                        min_anchor_count=min_anchor_count,
                        scope=f"Embedding space {previous_period!r} -> {period!r}",
                    )
                    anchor_count = _selected_anchor_count(
                        shared_count=anchor_count,
                        vector_size=previous_vectors.vector_size,
                        top_k_anchor=top_k_anchor,
                        min_anchor_count=min_anchor_count,
                    )
            results.append(
                _build_alignment_result(
                    period=period,
                    input_path=input_path,
                    paths=paths,
                    output_path=_primary_raw_vector_output_path(paths, save_formats),
                    aligned_to_period=previous_period,
                    anchor_count=anchor_count,
                    vocabulary_size=len(current_vectors),
                )
            )
            previous_vectors = current_vectors
            previous_frequencies = current_frequencies
            previous_period = period
            continue

        current_vectors = KeyedVectors.load_word2vec_format(
            str(input_path),
            binary=False,
        )
        _normalize_keyed_vectors(current_vectors)

        anchor_count = 0
        if previous_vectors is not None:
            if dependency_alignment_mode == "global":
                anchors = _select_dependency_anchors(
                    base_vectors=previous_vectors,
                    other_vectors=current_vectors,
                    base_frequencies=previous_frequencies,
                    other_frequencies=current_frequencies,
                    top_k_anchor=top_k_anchor,
                    min_anchor_count=min_anchor_count,
                    warning_scope=(
                        f"Dependency embedding space "
                        f"{previous_period!r} -> {period!r}"
                    ),
                )
                rotation, anchor_count = _orthogonal_procrustes_rotation_from_anchors(
                    base_vectors=previous_vectors,
                    other_vectors=current_vectors,
                    anchors=anchors,
                )
                current_vectors.vectors = current_vectors.vectors.dot(rotation).astype(
                    np.float32,
                    copy=False,
                )
            elif dependency_alignment_mode == "region_specific":
                anchors_by_relation = _select_dependency_anchors_by_relation(
                    base_vectors=previous_vectors,
                    other_vectors=current_vectors,
                    base_frequencies=previous_frequencies,
                    other_frequencies=current_frequencies,
                    top_k_anchor_pct=top_k_anchor_pct,
                    min_anchor_count=min_anchor_count,
                    warning_scope=(
                        f"Dependency regions {previous_period!r} -> {period!r}"
                    ),
                )
                rotations, anchor_count = _dependency_region_rotations(
                    base_vectors=previous_vectors,
                    other_vectors=current_vectors,
                    anchors_by_relation=anchors_by_relation,
                )
                _apply_dependency_region_rotations(current_vectors, rotations)
            else:
                rotation, anchor_count = _orthogonal_procrustes_rotation_for_vectors(
                    base_vectors=previous_vectors,
                    other_vectors=current_vectors,
                    min_anchor_count=min_anchor_count,
                    top_k_anchor=top_k_anchor,
                    base_frequencies=previous_frequencies,
                    other_frequencies=current_frequencies,
                    warning_scope=(
                        f"Embedding space {previous_period!r} -> {period!r}"
                    ),
                )
                current_vectors.vectors = current_vectors.vectors.dot(rotation).astype(
                    np.float32,
                    copy=False,
                )
            _reset_keyed_vector_norms(current_vectors)

        paths["keyed_vectors"].parent.mkdir(parents=True, exist_ok=True)
        _save_aligned_keyed_vector_outputs(
            keyed_vectors=current_vectors,
            paths=paths,
            save_formats=save_formats,
            overwrite=overwrite,
        )
        results.append(
            _build_alignment_result(
                period=period,
                input_path=input_path,
                paths=paths,
                output_path=_primary_raw_vector_output_path(paths, save_formats),
                aligned_to_period=previous_period,
                anchor_count=anchor_count,
                vocabulary_size=len(current_vectors),
            )
        )
        previous_vectors = current_vectors
        previous_frequencies = current_frequencies
        previous_period = period

    return results


def _discover_period_subfolders(input_root: Path) -> list[str]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_root}")
    return sorted(path.name for path in input_root.iterdir() if path.is_dir())


def _resolve_period_model_path(
    input_root: Path,
    period: str | int,
    model_filename: str,
) -> Path:
    period_name = str(period)
    path = input_root / period_name / model_filename.format(period=period_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing W2V model for period {period}: {path}")
    return path


def _resolve_period_vector_path(
    input_root: Path,
    period: str | int,
    vector_filename: str,
) -> Path:
    period_name = str(period)
    path = input_root / period_name / vector_filename.format(period=period_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing W2V vector file for period {period}: {path}")
    return path


def _resolve_period_vocab_frequency_path(
    input_root: Path,
    period: str | int,
    filename: str,
) -> Path:
    period_name = str(period)
    path = input_root / period_name / filename.format(period=period_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing W2V vocabulary frequency file for period {period}: {path}"
        )
    return path


def _load_vocab_frequencies(path: Path) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    is_csv = path.suffix.lower() == ".csv"
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = (
            csv.reader(file)
            if is_csv
            else csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        )
        if is_csv and next(reader, None) != ["vocab", "frequency"]:
            raise ValueError(f"Expected vocab,frequency header in {path}")
        for line_number, row in enumerate(reader, start=2 if is_csv else 1):
            if not row:
                continue
            if len(row) != 2 or not row[0]:
                raise ValueError(f"Invalid vocabulary frequency row in {path}:{line_number}")
            word, count_text = row
            try:
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid vocabulary frequency in {path}:{line_number}"
                ) from exc
            if count < 1 or word in frequencies:
                raise ValueError(f"Invalid or duplicate frequency in {path}:{line_number}")
            frequencies[word] = count
    return frequencies


def _build_alignment_output_paths(
    output_root: Path,
    period: str | int,
) -> dict[str, Path]:
    period_name = str(period)
    output_dir = output_root / period_name
    output_stem = output_dir / period_name
    return {
        "model": output_stem.with_suffix(".model"),
        "keyed_vectors": output_stem.with_suffix(".kv"),
        "vectors_bin": output_stem.with_name(f"{period_name}_vectors.bin"),
        "vectors_txt": output_stem.with_name(f"{period_name}_vectors.txt"),
    }


def _normalize_keyed_vectors(keyed_vectors) -> None:
    keyed_vectors.vectors = _normalize_matrix(keyed_vectors.vectors).astype(
        np.float32,
        copy=False,
    )
    _reset_keyed_vector_norms(keyed_vectors)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _reset_keyed_vector_norms(keyed_vectors) -> None:
    if hasattr(keyed_vectors, "norms"):
        keyed_vectors.norms = None
    if hasattr(keyed_vectors, "fill_norms"):
        keyed_vectors.fill_norms(force=True)


def _orthogonal_procrustes_rotation(
    *,
    base_model: Word2Vec,
    other_model: Word2Vec,
    min_anchor_count: int | None,
    top_k_anchor: int | None,
    warning_scope: str,
) -> tuple[np.ndarray, int]:
    return _orthogonal_procrustes_rotation_for_vectors(
        base_vectors=base_model.wv,
        other_vectors=other_model.wv,
        min_anchor_count=min_anchor_count,
        top_k_anchor=top_k_anchor,
        warning_scope=warning_scope,
    )


def _orthogonal_procrustes_rotation_for_vectors(
    *,
    base_vectors: KeyedVectors,
    other_vectors: KeyedVectors,
    min_anchor_count: int | None,
    top_k_anchor: int | None = None,
    base_frequencies: Mapping[str, int] | None = None,
    other_frequencies: Mapping[str, int] | None = None,
    warning_scope: str = "Embedding space",
) -> tuple[np.ndarray, int]:
    anchors = sorted(set(base_vectors.key_to_index) & set(other_vectors.key_to_index))
    _warn_if_below_minimum_anchor_count(
        available_count=len(anchors),
        vector_size=base_vectors.vector_size,
        min_anchor_count=min_anchor_count,
        scope=warning_scope,
    )
    if top_k_anchor is not None:
        if base_frequencies is not None and other_frequencies is not None:
            missing_word = next(
                (
                    word for word in anchors
                    if word not in base_frequencies or word not in other_frequencies
                ),
                None,
            )
            if missing_word is not None:
                raise ValueError(
                    f"Missing vocabulary frequency for shared word {missing_word!r}."
                )
            anchors.sort(
                key=lambda word: (
                    -min(
                        base_frequencies[word],
                        other_frequencies[word],
                    ),
                    word,
                )
            )
        elif base_frequencies is None and other_frequencies is None:
            try:
                anchors.sort(
                    key=lambda word: (
                        -min(
                            base_vectors.get_vecattr(word, "count"),
                            other_vectors.get_vecattr(word, "count"),
                        ),
                        word,
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    "top_k_anchor requires word counts in both Word2Vec models."
                ) from exc
        else:
            raise ValueError("top_k_anchor requires frequencies for both periods.")
        selected_count = _selected_anchor_count(
            shared_count=len(anchors),
            vector_size=base_vectors.vector_size,
            top_k_anchor=top_k_anchor,
            min_anchor_count=min_anchor_count,
        )
        anchors = anchors[:selected_count]
    return _orthogonal_procrustes_rotation_from_anchors(
        base_vectors=base_vectors,
        other_vectors=other_vectors,
        anchors=anchors,
    )


def _selected_anchor_count(
    *,
    shared_count: int,
    vector_size: int,
    top_k_anchor: int | None,
    min_anchor_count: int | None,
) -> int:
    if top_k_anchor is None:
        return shared_count
    minimum = _minimum_anchor_count(vector_size, min_anchor_count)
    return min(shared_count, max(top_k_anchor, minimum))


def _minimum_anchor_count(
    vector_size: int,
    min_anchor_count: int | None,
) -> int:
    return vector_size if min_anchor_count is None else min_anchor_count


def _warn_if_below_minimum_anchor_count(
    *,
    available_count: int,
    vector_size: int,
    min_anchor_count: int | None,
    scope: str,
) -> None:
    minimum = _minimum_anchor_count(vector_size, min_anchor_count)
    if available_count >= minimum:
        return
    warnings.warn(
        f"{scope} has only {available_count} shared anchor candidates, below "
        f"the minimum anchor count of {minimum}. Using all available anchors "
        "and continuing alignment.",
        UserWarning,
        stacklevel=2,
    )


def _select_dependency_anchors(
    *,
    base_vectors: KeyedVectors,
    other_vectors: KeyedVectors,
    base_frequencies: Mapping[str, int] | None,
    other_frequencies: Mapping[str, int] | None,
    top_k_anchor: int | None,
    min_anchor_count: int | None,
    warning_scope: str = "Dependency embedding space",
) -> list[str]:
    if top_k_anchor is None:
        raise ValueError("Dependency alignment requires top_k_anchor.")
    if base_frequencies is None or other_frequencies is None:
        raise ValueError("Dependency alignment requires frequencies for both periods.")
    relation_items = _shared_dependency_items_by_relation(
        base_vectors=base_vectors,
        other_vectors=other_vectors,
        base_frequencies=base_frequencies,
        other_frequencies=other_frequencies,
    )
    shared_count = sum(map(len, relation_items.values()))
    _warn_if_below_minimum_anchor_count(
        available_count=shared_count,
        vector_size=base_vectors.vector_size,
        min_anchor_count=min_anchor_count,
        scope=warning_scope,
    )

    base_relation_totals = _relation_frequency_totals(base_frequencies)
    other_relation_totals = _relation_frequency_totals(other_frequencies)
    relations = sorted(set(base_relation_totals) | set(other_relation_totals))
    relation_support = {
        relation: min(
            base_relation_totals.get(relation, 0),
            other_relation_totals.get(relation, 0),
        )
        for relation in relations
    }

    total_support = sum(relation_support.values())
    if total_support < 1:
        raise ValueError("Dependency anchor candidates have no positive frequency support.")

    anchors: list[str] = []
    for relation in relations:
        support = relation_support[relation]
        quota = (2 * top_k_anchor * support + total_support) // (2 * total_support)
        ranked_items = sorted(
            relation_items.get(relation, []),
            key=lambda item: (
                -min(base_frequencies[item], other_frequencies[item]),
                item,
            ),
        )
        anchors.extend(ranked_items[:quota])

    minimum = _minimum_anchor_count(base_vectors.vector_size, min_anchor_count)
    target_count = min(shared_count, minimum)
    if len(anchors) < target_count:
        selected = set(anchors)
        remaining_items = sorted(
            (
                item
                for items in relation_items.values()
                for item in items
                if item not in selected
            ),
            key=lambda item: (
                -min(base_frequencies[item], other_frequencies[item]),
                item,
            ),
        )
        anchors.extend(remaining_items[: target_count - len(anchors)])
    return anchors


def _select_dependency_anchors_by_relation(
    *,
    base_vectors: KeyedVectors,
    other_vectors: KeyedVectors,
    base_frequencies: Mapping[str, int] | None,
    other_frequencies: Mapping[str, int] | None,
    top_k_anchor_pct: float | None,
    min_anchor_count: int | None,
    warning_scope: str = "Dependency regions",
) -> dict[str, list[str]]:
    if top_k_anchor_pct is None or not 0 < top_k_anchor_pct <= 100:
        raise ValueError(
            "Dependency region alignment requires top_k_anchor_pct in (0, 100]."
        )
    if base_frequencies is None or other_frequencies is None:
        raise ValueError("Dependency alignment requires frequencies for both periods.")
    relation_items = _shared_dependency_items_by_relation(
        base_vectors=base_vectors,
        other_vectors=other_vectors,
        base_frequencies=base_frequencies,
        other_frequencies=other_frequencies,
    )

    anchors_by_relation: dict[str, list[str]] = {}
    minimum = _minimum_anchor_count(base_vectors.vector_size, min_anchor_count)
    for relation, items in sorted(relation_items.items()):
        _warn_if_below_minimum_anchor_count(
            available_count=len(items),
            vector_size=base_vectors.vector_size,
            min_anchor_count=min_anchor_count,
            scope=f"{warning_scope}, relation {relation!r}",
        )
        percentage_count = max(
            1,
            math.ceil(len(items) * top_k_anchor_pct / 100),
        )
        anchor_count = min(len(items), max(percentage_count, minimum))
        anchors_by_relation[relation] = sorted(
            items,
            key=lambda item: (
                -min(base_frequencies[item], other_frequencies[item]),
                item,
            ),
        )[:anchor_count]
    return anchors_by_relation


def _shared_dependency_items_by_relation(
    *,
    base_vectors: KeyedVectors,
    other_vectors: KeyedVectors,
    base_frequencies: Mapping[str, int] | None,
    other_frequencies: Mapping[str, int] | None,
) -> dict[str, list[str]]:
    if base_frequencies is None or other_frequencies is None:
        raise ValueError("Dependency alignment requires frequencies for both periods.")

    shared_items = sorted(
        set(base_vectors.key_to_index) & set(other_vectors.key_to_index)
    )
    relation_items: dict[str, list[str]] = {}
    for item in shared_items:
        if item not in base_frequencies or item not in other_frequencies:
            raise ValueError(f"Missing vocabulary frequency for shared item {item!r}.")
        relation = _dependency_relation(item)
        relation_items.setdefault(relation, []).append(item)
    return relation_items


def _dependency_region_rotations(
    *,
    base_vectors: KeyedVectors,
    other_vectors: KeyedVectors,
    anchors_by_relation: Mapping[str, list[str]],
) -> tuple[dict[str, np.ndarray], int]:
    rotations: dict[str, np.ndarray] = {}
    anchor_count = 0
    for relation, anchors in anchors_by_relation.items():
        try:
            rotation, relation_anchor_count = (
                _orthogonal_procrustes_rotation_from_anchors(
                    base_vectors=base_vectors,
                    other_vectors=other_vectors,
                    anchors=anchors,
                )
            )
        except ValueError as exc:
            raise ValueError(f"Cannot align dependency relation {relation!r}: {exc}") from exc
        rotations[relation] = rotation
        anchor_count += relation_anchor_count
    return rotations, anchor_count


def _apply_dependency_region_rotations(
    keyed_vectors: KeyedVectors,
    rotations: Mapping[str, np.ndarray],
) -> None:
    indices_by_relation: dict[str, list[int]] = {}
    for index, item in enumerate(keyed_vectors.index_to_key):
        relation = _dependency_relation(item)
        indices_by_relation.setdefault(relation, []).append(index)

    missing_relations = sorted(set(indices_by_relation) - set(rotations))
    if missing_relations:
        joined = ", ".join(missing_relations)
        raise ValueError(
            "No shared anchors available for current-period dependency relation(s): "
            f"{joined}"
        )

    for relation, indices in indices_by_relation.items():
        keyed_vectors.vectors[indices] = keyed_vectors.vectors[indices].dot(
            rotations[relation]
        ).astype(np.float32, copy=False)


def _relation_frequency_totals(frequencies: Mapping[str, int]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item, frequency in frequencies.items():
        relation = _dependency_relation(item)
        totals[relation] = totals.get(relation, 0) + frequency
    return totals


def _dependency_relation(item: str) -> str:
    _, separator, relation = item.rpartition("/")
    if not separator or not relation:
        raise ValueError(
            f"Dependency vector item must use 'item/relation' format: {item!r}"
        )
    return relation


def _orthogonal_procrustes_rotation_from_anchors(
    *,
    base_vectors: KeyedVectors,
    other_vectors: KeyedVectors,
    anchors: list[str],
) -> tuple[np.ndarray, int]:
    if not anchors:
        raise ValueError("No shared vocabulary available for alignment.")

    base_indices = [base_vectors.key_to_index[word] for word in anchors]
    other_indices = [other_vectors.key_to_index[word] for word in anchors]
    base_matrix = base_vectors.vectors[base_indices]
    other_matrix = other_vectors.vectors[other_indices]

    matrix = other_matrix.T.dot(base_matrix)
    u_matrix, _, vt_matrix = np.linalg.svd(matrix)
    rotation = u_matrix.dot(vt_matrix)
    return rotation.astype(np.float32), len(anchors)


def _count_common_vocab(base_model: Word2Vec, other_model: Word2Vec) -> int:
    return len(set(base_model.wv.key_to_index) & set(other_model.wv.key_to_index))


def _count_common_keyed_vectors(
    base_vectors: KeyedVectors,
    other_vectors: KeyedVectors,
) -> int:
    return len(set(base_vectors.key_to_index) & set(other_vectors.key_to_index))


def _save_aligned_model_outputs(
    *,
    model: Word2Vec,
    paths: dict[str, Path],
    save_formats: tuple[str, ...],
    overwrite: bool,
) -> None:
    if "model" in save_formats and (overwrite or not paths["model"].exists()):
        model.save(str(paths["model"]))
    if "keyed_vectors" in save_formats and (
        overwrite or not paths["keyed_vectors"].exists()
    ):
        model.wv.save(str(paths["keyed_vectors"]))
    if "vectors_bin" in save_formats and (overwrite or not paths["vectors_bin"].exists()):
        model.wv.save_word2vec_format(str(paths["vectors_bin"]), binary=True)
    if "vectors_txt" in save_formats and (overwrite or not paths["vectors_txt"].exists()):
        model.wv.save_word2vec_format(str(paths["vectors_txt"]), binary=False)


def _save_aligned_keyed_vector_outputs(
    *,
    keyed_vectors: KeyedVectors,
    paths: dict[str, Path],
    save_formats: tuple[str, ...],
    overwrite: bool,
) -> None:
    if "keyed_vectors" in save_formats and (
        overwrite or not paths["keyed_vectors"].exists()
    ):
        keyed_vectors.save(str(paths["keyed_vectors"]))
    if "vectors_bin" in save_formats and (overwrite or not paths["vectors_bin"].exists()):
        keyed_vectors.save_word2vec_format(str(paths["vectors_bin"]), binary=True)
    if "vectors_txt" in save_formats and (overwrite or not paths["vectors_txt"].exists()):
        keyed_vectors.save_word2vec_format(str(paths["vectors_txt"]), binary=False)


def _find_existing_raw_vector_output(
    paths: dict[str, Path],
    save_formats: tuple[str, ...],
) -> Path | None:
    for format_name in ("keyed_vectors", "vectors_bin", "vectors_txt"):
        if format_name in save_formats and paths[format_name].exists():
            return paths[format_name]
    return None


def _load_existing_raw_vector_output(
    path: Path,
    paths: dict[str, Path],
) -> KeyedVectors:
    if path == paths["keyed_vectors"]:
        return KeyedVectors.load(str(path))
    if path == paths["vectors_bin"]:
        return KeyedVectors.load_word2vec_format(str(path), binary=True)
    return KeyedVectors.load_word2vec_format(str(path), binary=False)


def _primary_raw_vector_output_path(
    paths: dict[str, Path],
    save_formats: tuple[str, ...],
) -> Path:
    for format_name in ("keyed_vectors", "vectors_bin", "vectors_txt"):
        if format_name in save_formats:
            return paths[format_name]

    raise ValueError("save_formats must contain at least one raw vector format.")


def _build_alignment_result(
    *,
    period: str | int,
    input_path: Path,
    paths: dict[str, Path],
    output_path: Path | None = None,
    aligned_to_period: str | int | None,
    anchor_count: int,
    vocabulary_size: int,
) -> W2VAlignmentResult:
    return W2VAlignmentResult(
        period=period,
        input_path=input_path,
        output_path=paths["model"] if output_path is None else output_path,
        keyed_vectors_path=paths["keyed_vectors"],
        vectors_bin_path=paths["vectors_bin"],
        vectors_txt_path=paths["vectors_txt"],
        aligned_to_period=aligned_to_period,
        anchor_count=anchor_count,
        vocabulary_size=vocabulary_size,
    )
