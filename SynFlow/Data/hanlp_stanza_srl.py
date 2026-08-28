"""Parse raw sentence files with HanLP SRL and Stanza component lemmatisation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ALLOWED_INPUT_EXTENSIONS = frozenset({".txt", ".conll", ".conllu", ".json"})
_DEFAULT_OUTPUT_SUFFIX = ".txt"


def load_sentences(path: Path) -> list[str]:
    """Load non-empty sentences while preserving source line order."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def span_start(span: Any, fallback_index: int) -> int:
    """Return HanLP span start offset when available."""
    if len(span) >= 3 and isinstance(span[2], int):
        return span[2]
    return fallback_index


def lemmatise_component(srl_component: str, lemma_parser: Any) -> str:
    """Lemmatise one SRL component by parsing it with Stanza."""
    doc = lemma_parser(srl_component)
    lemmas = [word.lemma or word.text for sentence in doc.sentences for word in sentence.words]
    return " ".join(lemmas) if lemmas else srl_component


def frame_to_rows(
    frame: list[Any],
    lemma_parser: Any,
) -> list[tuple[str, str, int, int, str]]:
    """Convert one HanLP SRL frame into output rows.

    Each returned row contains SRL component text, lemmatised SRL component
    text, component id, head id, and SRL relation. Arguments point to the
    predicate component id; the predicate points to head id 0.
    """
    sorted_frame = sorted(enumerate(frame), key=lambda item: span_start(item[1], item[0]))
    predicate_component_id: int | None = None

    for component_id, (_, span) in enumerate(sorted_frame, start=1):
        srl_relation = str(span[1])
        if srl_relation == "PRED":
            predicate_component_id = component_id
            break

    if predicate_component_id is None:
        return []

    rows: list[tuple[str, str, int, int, str]] = []
    for component_id, (_, span) in enumerate(sorted_frame, start=1):
        srl_component = str(span[0])
        lemmatised_srl_component = lemmatise_component(srl_component, lemma_parser)
        srl_relation = str(span[1])
        head_id = 0 if srl_relation == "PRED" else predicate_component_id
        rows.append(
            (
                srl_component,
                lemmatised_srl_component,
                component_id,
                head_id,
                srl_relation,
            )
        )

    return rows


def file_id_for_output(
    input_file: Path,
    input_root: Path | None = None,
) -> str:
    """Build the file id used in <id=...>."""
    return input_file.stem


def output_file_for_input(
    input_file: Path,
    input_root: Path | None,
    output_path: Path,
    output_suffix: str | None = None,
) -> Path:
    """Return the output file path for one input file."""
    suffix = output_suffix if output_suffix is not None else _DEFAULT_OUTPUT_SUFFIX
    if input_root is None:
        if output_path.suffix:
            return output_path
        return output_path / f"{input_file.stem}{suffix}"

    relative_path = input_file.relative_to(input_root)
    return (output_path / relative_path).with_suffix(suffix)


def has_allowed_input_extension(
    path: Path,
    allowed_input_extensions: frozenset[str] = ALLOWED_INPUT_EXTENSIONS,
) -> bool:
    """Return whether a path satisfies the input file extension contract."""
    return path.suffix.lower() in allowed_input_extensions


def iter_input_files(
    input_path: Path,
    allowed_input_extensions: frozenset[str] = ALLOWED_INPUT_EXTENSIONS,
) -> list[Path]:
    """Return valid input files in deterministic relative path order."""
    if input_path.is_file():
        if not has_allowed_input_extension(input_path, allowed_input_extensions):
            allowed = ", ".join(sorted(allowed_input_extensions))
            raise ValueError(f"Input file extension must be one of: {allowed}. Got: {input_path}")
        return [input_path]

    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and has_allowed_input_extension(path, allowed_input_extensions)
        )

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def parse_hanlp_srl_batches(
    sentences: list[str],
    srl_parser: Any,
    hanlp_batch_size: int | None = None,
) -> list[Any]:
    """Run HanLP SRL in sentence batches and return SRL frames per sentence."""
    if hanlp_batch_size is None:
        return list(srl_parser(sentences)["srl"])
    if hanlp_batch_size <= 0:
        raise ValueError(f"hanlp_batch_size must be positive or None. Got: {hanlp_batch_size}")

    all_srl_frames: list[Any] = []
    for start in range(0, len(sentences), hanlp_batch_size):
        batch = sentences[start : start + hanlp_batch_size]
        all_srl_frames.extend(srl_parser(batch)["srl"])
    return all_srl_frames


def write_conll_like(
    input_file: Path,
    srl_parser: Any,
    lemma_parser: Any,
    output_file: Path,
    input_root: Path | None = None,
    hanlp_batch_size: int | None = None,
) -> int:
    """Run HanLP SRL, lemmatise SRL components with Stanza, and write output."""
    sentences = load_sentences(input_file)
    all_srl_frames = parse_hanlp_srl_batches(sentences, srl_parser, hanlp_batch_size)
    file_id = file_id_for_output(input_file, input_root)

    lines: list[str] = []
    for sentence_index, sentence_frames in enumerate(all_srl_frames, start=1):
        predicate_index = 0

        for frame in sentence_frames:
            rows = frame_to_rows(frame, lemma_parser)
            if not rows:
                continue

            predicate_index += 1
            lines.append(f"<id={file_id}_{sentence_index}_{predicate_index}>")

            for (
                srl_component,
                lemmatised_srl_component,
                component_id,
                head_id,
                srl_relation,
            ) in rows:
                lines.append(
                    f"{srl_component}\t{lemmatised_srl_component}\t-\t"
                    f"{component_id}\t{head_id}\t{srl_relation}\t-"
                )

            lines.append("<s>")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(sentences)


def build_hanlp_srl_parser(model_name: str, gpu_id: int | None = None) -> Any:
    """Load the HanLP parser used for semantic role labelling."""
    import hanlp

    if gpu_id is None:
        return hanlp.load(model_name)
    return hanlp.load(model_name, devices=gpu_id)


def build_stanza_pipeline(
    lang: str,
    package: str | dict[str, str] | None = None,
    gpu_id: int | None = None,
    processors: str | dict[str, str] = "tokenize,pos,lemma",
    **kwargs: Any,
) -> Any:
    """Build a Stanza pipeline from language-only or explicit package config."""
    import stanza

    pipeline_kwargs: dict[str, Any] = {
        "lang": lang,
        "processors": processors,
        "tokenize_no_ssplit": True,
    }
    if package is not None:
        pipeline_kwargs["package"] = package
    if gpu_id is not None:
        pipeline_kwargs["device"] = f"cuda:{gpu_id}"
        pipeline_kwargs["use_gpu"] = True
    pipeline_kwargs.update(kwargs)
    return stanza.Pipeline(**pipeline_kwargs)


def parse_path(
    input_path: Path,
    output_path: Path,
    srl_parser: Any,
    lemma_parser: Any,
    allowed_input_extensions: frozenset[str] = ALLOWED_INPUT_EXTENSIONS,
    output_suffix: str | None = None,
    hanlp_batch_size: int | None = None,
    verbose: bool = True,
) -> dict[Path, int]:
    """Parse one file or a folder tree and keep folder structure in output."""
    input_root = input_path if input_path.is_dir() else None
    input_files = iter_input_files(input_path, allowed_input_extensions)
    if not input_files:
        allowed = ", ".join(sorted(allowed_input_extensions))
        raise FileNotFoundError(f"No input files with extensions {allowed} under {input_path}")

    sentence_counts: dict[Path, int] = {}
    for input_file in input_files:
        output_file = output_file_for_input(input_file, input_root, output_path, output_suffix)
        sentence_count = write_conll_like(
            input_file=input_file,
            srl_parser=srl_parser,
            lemma_parser=lemma_parser,
            output_file=output_file,
            input_root=input_root,
            hanlp_batch_size=hanlp_batch_size,
        )
        sentence_counts[input_file] = sentence_count
        if verbose:
            print(f"Parsed {sentence_count} sentences: {input_file} -> {output_file}")

    if verbose:
        total_sentences = sum(sentence_counts.values())
        print(f"Done. Parsed {len(sentence_counts)} files and {total_sentences} sentences.")

    return sentence_counts


def split_round_robin(items: list[Path], chunk_count: int) -> list[list[Path]]:
    """Split items into non-empty round-robin chunks."""
    chunks: list[list[Path]] = [[] for _ in range(chunk_count)]
    for index, item in enumerate(items):
        chunks[index % chunk_count].append(item)
    return [chunk for chunk in chunks if chunk]


def _parse_file_chunk(
    input_files: list[Path],
    input_root: Path | None,
    output_path: Path,
    output_suffix: str | None,
    hanlp_model_name: str,
    stanza_lang: str,
    stanza_package: str | dict[str, str] | None,
    stanza_processors: str | dict[str, str],
    stanza_kwargs: dict[str, Any],
    gpu_id: int,
    hanlp_batch_size: int | None,
) -> list[tuple[Path, Path, int, int]]:
    """Worker entrypoint: load parsers on one GPU and parse a file chunk."""
    srl_parser = build_hanlp_srl_parser(hanlp_model_name, gpu_id=gpu_id)
    lemma_parser = build_stanza_pipeline(
        lang=stanza_lang,
        package=stanza_package,
        gpu_id=gpu_id,
        processors=stanza_processors,
        **stanza_kwargs,
    )

    results: list[tuple[Path, Path, int, int]] = []
    for input_file in input_files:
        output_file = output_file_for_input(input_file, input_root, output_path, output_suffix)
        sentence_count = write_conll_like(
            input_file=input_file,
            srl_parser=srl_parser,
            lemma_parser=lemma_parser,
            output_file=output_file,
            input_root=input_root,
            hanlp_batch_size=hanlp_batch_size,
        )
        results.append((input_file, output_file, sentence_count, gpu_id))
    return results


def hanlp_stanza_parse_folder(
    input_path: Path,
    output_path: Path,
    hanlp_model_name: str,
    stanza_lang: str,
    stanza_package: str | dict[str, str] | None,
    gpu_ids: list[int],
    num_worker_per_gpu: int,
    stanza_processors: str | dict[str, str] = "tokenize,pos,lemma",
    stanza_kwargs: dict[str, Any] | None = None,
    allowed_input_extensions: frozenset[str] = ALLOWED_INPUT_EXTENSIONS,
    output_suffix: str | None = None,
    hanlp_batch_size: int | None = None,
    verbose: bool = True,
) -> dict[Path, int]:
    """Parse a file/folder tree with HanLP SRL and Stanza lemmatisation.

    HanLP is batched by sentence count through hanlp_batch_size. Stanza is kept
    component-by-component inside each worker to avoid sending all SRL
    components from a sentence/file as one large Stanza batch.
    """
    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one GPU id.")
    if num_worker_per_gpu <= 0:
        raise ValueError(f"num_worker_per_gpu must be positive. Got: {num_worker_per_gpu}")
    if hanlp_batch_size is not None and hanlp_batch_size <= 0:
        raise ValueError(f"hanlp_batch_size must be positive or None. Got: {hanlp_batch_size}")

    input_root = input_path if input_path.is_dir() else None
    input_files = iter_input_files(input_path, allowed_input_extensions)
    if not input_files:
        allowed = ", ".join(sorted(allowed_input_extensions))
        raise FileNotFoundError(f"No input files with extensions {allowed} under {input_path}")

    max_workers = min(len(input_files), len(gpu_ids) * num_worker_per_gpu)
    chunks = split_round_robin(input_files, max_workers)
    worker_gpu_ids = [gpu_ids[index % len(gpu_ids)] for index in range(len(chunks))]
    stanza_options = stanza_kwargs or {}

    sentence_counts: dict[Path, int] = {}
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _parse_file_chunk,
                chunk,
                input_root,
                output_path,
                output_suffix,
                hanlp_model_name,
                stanza_lang,
                stanza_package,
                stanza_processors,
                stanza_options,
                gpu_id,
                hanlp_batch_size,
            )
            for chunk, gpu_id in zip(chunks, worker_gpu_ids)
        ]

        for future in as_completed(futures):
            for input_file, output_file, sentence_count, gpu_id in future.result():
                sentence_counts[input_file] = sentence_count
                if verbose:
                    print(
                        f"GPU {gpu_id}: parsed {sentence_count} sentences: "
                        f"{input_file} -> {output_file}"
                    )

    if verbose:
        total_sentences = sum(sentence_counts.values())
        print(f"Done. Parsed {len(sentence_counts)} files and {total_sentences} sentences.")

    return sentence_counts
