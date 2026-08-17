"""Core Word2Vec training utilities for period-split sentence folders."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from tqdm import tqdm
from typing import Iterable
import multiprocessing
import random
import tempfile

from gensim import utils
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

DEFAULT_SAVE_FORMATS = ("model", "keyed_vectors", "vectors_bin", "vectors_txt")
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

    return tqdm(iterable, total=total, desc="Training W2V", unit="period")


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
    show_progress: bool = True,
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
        min_anchor_count: Minimum shared vocabulary size required for aligning
            a period to the previous aligned period. If ``None``, the required
            anchor count is the embedding dimensionality.
        show_progress: Whether to show tqdm progress over periods.
        overwrite: Whether to replace existing aligned output models.
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

    results: list[W2VAlignmentResult] = []
    previous_model: Word2Vec | None = None
    previous_period: str | int | None = None

    iterator = _progress_iter(periods, total=len(periods), enabled=show_progress)
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
            results.append(
                _build_alignment_result(
                    period=period,
                    input_path=input_path,
                    paths=paths,
                    aligned_to_period=previous_period,
                    anchor_count=0 if previous_model is None else _count_common_vocab(
                        previous_model,
                        current_model,
                    ),
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
) -> tuple[np.ndarray, int]:
    anchors = sorted(set(base_model.wv.key_to_index) & set(other_model.wv.key_to_index))
    required_anchors = (
        base_model.wv.vector_size
        if min_anchor_count is None
        else min_anchor_count
    )
    if len(anchors) < required_anchors:
        raise ValueError(
            "Not enough shared vocabulary for alignment: "
            f"found {len(anchors)}, required {required_anchors}."
        )

    base_indices = [base_model.wv.key_to_index[word] for word in anchors]
    other_indices = [other_model.wv.key_to_index[word] for word in anchors]
    base_vectors = base_model.wv.vectors[base_indices]
    other_vectors = other_model.wv.vectors[other_indices]

    matrix = other_vectors.T.dot(base_vectors)
    u_matrix, _, vt_matrix = np.linalg.svd(matrix)
    rotation = u_matrix.dot(vt_matrix)
    return rotation.astype(np.float32), len(anchors)


def _count_common_vocab(base_model: Word2Vec, other_model: Word2Vec) -> int:
    return len(set(base_model.wv.key_to_index) & set(other_model.wv.key_to_index))


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


def _build_alignment_result(
    *,
    period: str | int,
    input_path: Path,
    paths: dict[str, Path],
    aligned_to_period: str | int | None,
    anchor_count: int,
    vocabulary_size: int,
) -> W2VAlignmentResult:
    return W2VAlignmentResult(
        period=period,
        input_path=input_path,
        output_path=paths["model"],
        keyed_vectors_path=paths["keyed_vectors"],
        vectors_bin_path=paths["vectors_bin"],
        vectors_txt_path=paths["vectors_txt"],
        aligned_to_period=aligned_to_period,
        anchor_count=anchor_count,
        vocabulary_size=vocabulary_size,
    )

