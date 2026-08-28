"""Parse raw sentence files with FrameSemanticTransformer and Stanza.

The input corpus is expected to contain raw sentence files inside subfolders
below an input root, where each line is one raw sentence.

The output mirrors the input directory structure and writes one frame block per
detected frame:

    <s id=PARENT_DIRECTORY_FILE_STEM_LINE_NUMBER_FRAME_NUMBER>
    component<TAB>lemmatised_component<TAB>-<TAB>component_id<TAB>head_id<TAB>relation<TAB>-
    </s>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, MutableMapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Iterable, Sequence

from tqdm import tqdm


FRAME_PIPELINE: Any | None = None
SMALL_FRAME_PIPELINE: Any | None = None
LEMMA_PIPELINE: Any | None = None
LEMMA_CACHE: MutableMapping[str, str] = {}
DEFAULT_FILE_EXTENSIONS = ("*.txt", "*.conll", "*.conllu", "*.json")
DEFAULT_STANZA_PROCESSORS = "tokenize,pos,lemma"
DEFAULT_FRAME_BATCH_SIZE = 24


@dataclass(frozen=True)
class ParseTask:
    """One input file and its mirrored output location."""

    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class ParseResult:
    """Summary for one parsed file."""

    input_path: Path
    output_path: Path
    sentence_count: int
    block_count: int
    skipped: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse raw sentence files into FrameSemanticTransformer plus "
            "Stanza lemma SynFlow format."
        )
    )
    parser.add_argument(
        "--input-root",
        "--input-dir",
        dest="input_root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Corpus root containing raw sentence files. "
            "Default: current directory."
        ),
    )
    parser.add_argument(
        "--output-root",
        "--output-dir",
        dest="output_root",
        type=Path,
        default=None,
        help=(
            "Output root. Default: sibling directory named "
            "<input-root-name>_frame_parsed."
        ),
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=(
            "File glob to parse recursively under the input root. "
            "Can be repeated. Default: *.txt, *.conll, *.conllu, *.json."
        ),
    )
    parser.add_argument(
        "--frame-model",
        default="base",
        help="FrameSemanticTransformer model name. Default: base.",
    )
    parser.add_argument(
        "--small-frame-model",
        default="small",
        help="Fallback FrameSemanticTransformer model name. Default: small.",
    )
    parser.add_argument(
        "--no-small-fallback",
        action="store_true",
        help="Disable fallback to the small frame model when the main model finds no frames.",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=DEFAULT_FRAME_BATCH_SIZE,
        help=(
            "Number of raw sentences sent to FrameSemanticTransformer in each "
            f"batch. Default: {DEFAULT_FRAME_BATCH_SIZE}."
        ),
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Required Stanza language code for component lemmatisation.",
    )
    stanza_package_group = parser.add_mutually_exclusive_group(required=True)
    stanza_package_group.add_argument(
        "--stanza-package",
        help=(
            "One Stanza package to use for all requested processors, "
            "for example ewt."
        ),
    )
    stanza_package_group.add_argument(
        "--stanza-package-json",
        help=(
            "JSON object mapping each Stanza processor to a package, for example "
            '\'{"tokenize":"ewt","pos":"ewt","lemma":"ewt"}\'.'
        ),
    )
    parser.add_argument(
        "--stanza-processors",
        default=DEFAULT_STANZA_PROCESSORS,
        help=(
            "Comma-separated Stanza processors for component lemmatisation. "
            f"Default: {DEFAULT_STANZA_PROCESSORS}."
        ),
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help=(
            "CUDA GPU id or comma-separated GPU ids to use, for example: "
            "--gpu 2 or --gpu 2,3. Default: 0."
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help=(
            "Number of worker processes to run on each selected GPU. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-parse files even when the output file already exists.",
    )
    return parser.parse_args()


def parse_stanza_package_json(raw_json: str | None) -> dict[str, str] | None:
    """Parse CLI JSON for Stanza per-processor packages."""
    if raw_json is None:
        return None

    stanza_package = json.loads(raw_json)
    if not isinstance(stanza_package, dict):
        raise ValueError("--stanza-package-json must be a JSON object")

    for processor, package in stanza_package.items():
        if not isinstance(processor, str) or not isinstance(package, str):
            raise ValueError("--stanza-package-json keys and values must be strings")

    return stanza_package


def parse_gpu_ids(gpu: str) -> list[int]:
    """Parse one or more CUDA GPU ids from the --gpu argument."""
    gpu_ids: list[int] = []
    for raw_gpu_id in gpu.split(","):
        gpu_id = raw_gpu_id.strip()
        if not gpu_id:
            raise ValueError("GPU ids must not be empty")
        if not gpu_id.isdigit():
            raise ValueError("GPU ids must be non-negative integers")
        gpu_ids.append(int(gpu_id))
    return gpu_ids


def build_worker_gpu_ids(gpu_ids: list[int], workers_per_gpu: int) -> list[int]:
    """Expand GPU ids so each selected GPU gets workers_per_gpu workers."""
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be at least 1")
    return [gpu_id for gpu_id in gpu_ids for _ in range(workers_per_gpu)]


def validate_parse_config(
    frame_model_name: str,
    small_frame_model_name: str,
    frame_batch_size: int,
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str,
) -> tuple[str, str, str, str | Mapping[str, str]]:
    """Validate required frame parser and Stanza configuration."""
    frame_model_name = frame_model_name.strip()
    small_frame_model_name = small_frame_model_name.strip()
    language = language.strip()
    stanza_processors = stanza_processors.strip()

    if not frame_model_name:
        raise ValueError("frame_model_name must be a non-empty model name")
    if not small_frame_model_name:
        raise ValueError("small_frame_model_name must be a non-empty model name")
    if frame_batch_size < 1:
        raise ValueError("frame_batch_size must be at least 1")
    if not language:
        raise ValueError("language must be a non-empty Stanza language code")
    if isinstance(stanza_package, str) and not stanza_package.strip():
        raise ValueError("stanza_package must be a non-empty Stanza package name")
    if isinstance(stanza_package, Mapping) and not stanza_package:
        raise ValueError("stanza_package must not be empty")
    if isinstance(stanza_package, Mapping):
        for processor, package in stanza_package.items():
            if not isinstance(processor, str) or not isinstance(package, str):
                raise ValueError("stanza_package keys and values must be strings")
    if not stanza_processors:
        raise ValueError("stanza_processors must not be empty")

    return frame_model_name, small_frame_model_name, language, stanza_package


def build_frame_pipeline(
    model_name: str,
    frame_batch_size: int,
    use_gpu: bool,
) -> Any:
    """Load one FrameSemanticTransformer model."""
    from frame_semantic_transformer import FrameSemanticTransformer

    print(
        f"Loading FrameSemanticTransformer with model={model_name!r}, "
        f"use_gpu={use_gpu}",
        flush=True,
    )
    transformer = FrameSemanticTransformer(
        model_name,
        batch_size=frame_batch_size,
        use_gpu=use_gpu,
    )
    transformer.setup()
    return transformer


def build_stanza_lemma_pipeline(
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str,
    use_gpu: bool,
) -> Any:
    """Load one Stanza lemmatisation pipeline."""
    import stanza
    from stanza.pipeline.core import DownloadMethod

    device = "cuda:0" if use_gpu else "cpu"
    resolved_package: str | dict[str, str] = (
        dict(stanza_package)
        if isinstance(stanza_package, Mapping)
        else stanza_package
    )
    print(
        f"Loading Stanza {language} on {device} "
        f"with package={resolved_package!r}, processors={stanza_processors!r}",
        flush=True,
    )

    return stanza.Pipeline(
        lang=language,
        package=resolved_package,
        processors=stanza_processors,
        tokenize_no_ssplit=True,
        use_gpu=use_gpu,
        device=device,
        download_method=DownloadMethod.REUSE_RESOURCES,
    )


def init_worker(
    gpu_queue: Queue[int],
    frame_model_name: str,
    small_frame_model_name: str,
    frame_batch_size: int,
    use_small_fallback: bool,
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str,
) -> None:
    """Load frame and Stanza pipelines on the GPU assigned to this worker."""
    global FRAME_PIPELINE, SMALL_FRAME_PIPELINE, LEMMA_PIPELINE, LEMMA_CACHE
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    FRAME_PIPELINE = build_frame_pipeline(
        frame_model_name,
        frame_batch_size=frame_batch_size,
        use_gpu=True,
    )
    SMALL_FRAME_PIPELINE = None
    if use_small_fallback:
        SMALL_FRAME_PIPELINE = build_frame_pipeline(
            small_frame_model_name,
            frame_batch_size=frame_batch_size,
            use_gpu=True,
        )
    LEMMA_PIPELINE = build_stanza_lemma_pipeline(
        language=language,
        stanza_package=stanza_package,
        stanza_processors=stanza_processors,
        use_gpu=True,
    )
    LEMMA_CACHE = {}


def discover_tasks(
    input_root: Path,
    output_root: Path,
    patterns: Sequence[str] = DEFAULT_FILE_EXTENSIONS,
) -> list[ParseTask]:
    """Return parse tasks for matching files inside input-root subfolders."""
    tasks: list[ParseTask] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for input_path in sorted(
            path for path in input_root.rglob(pattern) if path.is_file()
        ):
            relative_path = input_path.relative_to(input_root)
            if len(relative_path.parts) < 2 or input_path in seen:
                continue
            seen.add(input_path)
            tasks.append(
                ParseTask(
                    input_path=input_path,
                    output_path=output_root / relative_path,
                )
            )
    return tasks


def frame_stanza_parse_folder(
    input_root: str | Path,
    output_root: str | Path,
    *,
    frame_model_name: str = "base",
    small_frame_model_name: str = "small",
    frame_batch_size: int = DEFAULT_FRAME_BATCH_SIZE,
    use_small_fallback: bool = True,
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str = DEFAULT_STANZA_PROCESSORS,
    gpu: str = "0",
    workers_per_gpu: int = 1,
    overwrite: bool = False,
    file_patterns: Sequence[str] = DEFAULT_FILE_EXTENSIONS,
) -> Path:
    """Parse raw sentence files inside input-root subfolders."""
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    frame_model_name, small_frame_model_name, language, stanza_package = (
        validate_parse_config(
            frame_model_name=frame_model_name,
            small_frame_model_name=small_frame_model_name,
            frame_batch_size=frame_batch_size,
            language=language,
            stanza_package=stanza_package,
            stanza_processors=stanza_processors,
        )
    )

    tasks = discover_tasks(input_root, output_root, file_patterns)
    if not tasks:
        extensions = ", ".join(file_patterns)
        raise FileNotFoundError(
            f"No subfolder files matched {extensions} under {input_root}"
        )

    gpu_ids = parse_gpu_ids(gpu)
    worker_gpu_ids = build_worker_gpu_ids(gpu_ids, workers_per_gpu)

    print(f"Input root: {input_root}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"Files: {len(tasks)}", flush=True)
    print(
        f"Frame model: {frame_model_name}; "
        f"small fallback: {use_small_fallback}; "
        f"small frame model: {small_frame_model_name}; "
        f"Stanza language: {language}; "
        f"Stanza package: {stanza_package}; "
        f"Stanza processors: {stanza_processors}",
        flush=True,
    )
    print(
        f"GPUs: {', '.join(f'cuda:{gpu_id}' for gpu_id in gpu_ids)}; "
        f"workers per GPU: {workers_per_gpu}; "
        f"total workers: {len(worker_gpu_ids)}; "
        f"frame batch size: {frame_batch_size}; "
        f"overwrite: {overwrite}",
        flush=True,
    )

    run_tasks(
        tasks=tasks,
        frame_batch_size=frame_batch_size,
        worker_gpu_ids=worker_gpu_ids,
        overwrite=overwrite,
        frame_model_name=frame_model_name,
        small_frame_model_name=small_frame_model_name,
        use_small_fallback=use_small_fallback,
        language=language,
        stanza_package=stanza_package,
        stanza_processors=stanza_processors,
    )
    return output_root


def batched(
    items: list[tuple[str, str]],
    batch_size: int,
) -> Iterable[list[tuple[str, str]]]:
    """Yield fixed-size batches."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def read_sentence_rows(input_path: Path) -> list[tuple[str, str]]:
    """Read raw sentence lines from one input file."""
    rows: list[tuple[str, str]] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append((str(line_number), stripped))
    return rows


def sanitize_tsv_field(value: object) -> str:
    """Keep output records one-row-per-component by removing tabs and newlines."""
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def trigger_word_from_location(sentence: str, trigger_location: int) -> str:
    """Return the token beginning at a model-provided character offset."""
    if trigger_location < 0 or trigger_location >= len(sentence):
        return ""
    match = re.search(r"\S+", sentence[trigger_location:])
    return match.group(0) if match else ""


def detect_frames_bulk(sentences: list[str]) -> list[Any]:
    """Run bulk frame detection when available."""
    if FRAME_PIPELINE is None:
        raise RuntimeError("Frame pipeline was not initialized")
    if hasattr(FRAME_PIPELINE, "detect_frames_bulk"):
        return FRAME_PIPELINE.detect_frames_bulk(sentences)
    return [FRAME_PIPELINE.detect_frames(sentence) for sentence in sentences]


def detect_frames_with_fallback(sentence: str, base_result: Any) -> Any:
    """Return the base result, or small-model fallback result when configured."""
    if getattr(base_result, "frames", []) or []:
        return base_result
    if SMALL_FRAME_PIPELINE is None:
        return base_result
    return SMALL_FRAME_PIPELINE.detect_frames(sentence)


def lemmatise_component(component: str) -> str:
    """Lemmatise one frame component by parsing it with Stanza."""
    if not component:
        return "-"
    if LEMMA_PIPELINE is None:
        raise RuntimeError("Stanza lemma pipeline was not initialized")

    cache_key = sanitize_tsv_field(component)
    if cache_key in LEMMA_CACHE:
        return LEMMA_CACHE[cache_key]

    doc = LEMMA_PIPELINE(cache_key)
    lemmas = [
        word.lemma or word.text
        for sentence in doc.sentences
        for word in sentence.words
    ]
    lemma = sanitize_tsv_field(" ".join(lemmas)) or "-"
    LEMMA_CACHE[cache_key] = lemma
    return lemma


def sentence_id_base(input_path: Path, sentence_index: str) -> str:
    """Build the shared SynFlow sentence id base for one input line."""
    return f"{input_path.parent.name}_{input_path.stem}_{sentence_index}"


def format_component_row(
    component: str,
    lemma: str,
    component_id: int,
    head_id: int,
    relation: str,
) -> str:
    """Serialize one frame component row in SynFlow's seven-field format."""
    return (
        f"{component}\t{lemma}\t-\t{component_id}\t{head_id}\t"
        f"{relation}\t-"
    )


def format_frame_block(
    sentence_id: str,
    frame_index: int,
    result: Any,
    frame: Any,
) -> str:
    """Serialize one detected frame as one SynFlow block."""
    sentence = getattr(result, "sentence", "") or ""
    frame_name = sanitize_tsv_field(getattr(frame, "name", "")) or "-"
    trigger_location = getattr(frame, "trigger_location", -1)
    trigger = sanitize_tsv_field(
        trigger_word_from_location(sentence, trigger_location)
    )
    trigger_lemma = lemmatise_component(trigger)

    lines = [
        f"<s id={sentence_id}_{frame_index}>",
        format_component_row(frame_name, frame_name, 1, 0, "frame"),
        format_component_row(trigger or "-", trigger_lemma, 2, 1, "frametrigger"),
    ]

    frame_elements = getattr(frame, "frame_elements", []) or []
    for component_id, frame_element in enumerate(frame_elements, start=3):
        element_name = sanitize_tsv_field(getattr(frame_element, "name", "")) or "-"
        element_text = sanitize_tsv_field(getattr(frame_element, "text", "")) or "-"
        element_lemma = lemmatise_component(element_text)
        lines.append(
            format_component_row(
                element_text,
                element_lemma,
                component_id,
                1,
                element_name,
            )
        )

    lines.append("</s>")
    return "\n".join(lines)


def fsync_parent(path: Path) -> None:
    """Ensure a directory entry is durable after os.replace."""
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def parse_file(
    task: ParseTask,
    frame_batch_size: int,
    overwrite: bool,
) -> ParseResult:
    """Parse one corpus file and atomically publish the mirrored output."""
    if task.output_path.exists() and not overwrite:
        return ParseResult(task.input_path, task.output_path, 0, 0, skipped=True)

    rows = read_sentence_rows(task.input_path)
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = task.output_path.with_name(f".{task.output_path.name}.tmp")
    tmp_path.unlink(missing_ok=True)

    block_count = 0
    with tmp_path.open("w", encoding="utf-8") as output_file:
        for batch in batched(rows, frame_batch_size):
            sentence_indexes = [sentence_index for sentence_index, _ in batch]
            sentences = [sentence for _, sentence in batch]
            base_results = detect_frames_bulk(sentences)
            if len(base_results) != len(sentences):
                raise RuntimeError(
                    f"FrameSemanticTransformer returned {len(base_results)} "
                    f"results for {len(sentences)} inputs"
                )

            for sentence_index, sentence, base_result in zip(
                sentence_indexes,
                sentences,
                base_results,
            ):
                result = detect_frames_with_fallback(sentence, base_result)
                frames = getattr(result, "frames", []) or []
                for frame_index, frame in enumerate(frames, start=1):
                    output_file.write(
                        format_frame_block(
                            sentence_id=sentence_id_base(
                                task.input_path,
                                sentence_index,
                            ),
                            frame_index=frame_index,
                            result=result,
                            frame=frame,
                        )
                    )
                    output_file.write("\n")
                    block_count += 1

        output_file.flush()
        os.fsync(output_file.fileno())

    os.replace(tmp_path, task.output_path)
    fsync_parent(task.output_path)
    return ParseResult(
        task.input_path,
        task.output_path,
        len(rows),
        block_count,
        skipped=False,
    )


def run_tasks(
    tasks: list[ParseTask],
    frame_batch_size: int,
    worker_gpu_ids: list[int],
    overwrite: bool,
    frame_model_name: str,
    small_frame_model_name: str,
    use_small_fallback: bool,
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str,
) -> None:
    """Parse files in parallel with the requested GPU worker layout."""
    parsed_sentences = 0
    parsed_blocks = 0
    skipped_files = 0
    gpu_queue: Queue[int] = Queue()
    for gpu_id in worker_gpu_ids:
        gpu_queue.put(gpu_id)

    with ProcessPoolExecutor(
        max_workers=len(worker_gpu_ids),
        initializer=init_worker,
        initargs=(
            gpu_queue,
            frame_model_name,
            small_frame_model_name,
            frame_batch_size,
            use_small_fallback,
            language,
            stanza_package,
            stanza_processors,
        ),
    ) as executor:
        futures = [
            executor.submit(
                parse_file,
                task,
                frame_batch_size,
                overwrite,
            )
            for task in tasks
        ]
        with tqdm(total=len(futures), desc="Files", unit="file") as file_progress:
            for future in as_completed(futures):
                result = future.result()
                parsed_sentences += result.sentence_count
                parsed_blocks += result.block_count
                skipped_files += int(result.skipped)
                if result.skipped:
                    file_progress.set_postfix_str(
                        f"skipped: {result.input_path.name}"
                    )
                else:
                    file_progress.set_postfix_str(
                        f"parsed {result.block_count}: "
                        f"{result.input_path.name}"
                    )
                file_progress.update(1)

    print(
        f"Done. Files: {len(tasks)}, skipped: {skipped_files}, "
        f"sentences parsed: {parsed_sentences}, frame blocks: {parsed_blocks}",
        flush=True,
    )


def main() -> None:
    """Parse all discovered corpus files."""
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else input_root.parent / f"{input_root.name}_frame_parsed"
    )
    patterns = args.pattern if args.pattern is not None else DEFAULT_FILE_EXTENSIONS
    stanza_package = (
        parse_stanza_package_json(args.stanza_package_json)
        if args.stanza_package_json is not None
        else args.stanza_package
    )
    if stanza_package is None:
        raise ValueError("A Stanza package must be provided")

    frame_stanza_parse_folder(
        input_root=input_root,
        output_root=output_root,
        frame_model_name=args.frame_model,
        small_frame_model_name=args.small_frame_model,
        frame_batch_size=args.frame_batch_size,
        use_small_fallback=not args.no_small_fallback,
        language=args.language,
        stanza_package=stanza_package,
        stanza_processors=args.stanza_processors,
        gpu=args.gpu,
        workers_per_gpu=args.workers_per_gpu,
        overwrite=args.overwrite,
        file_patterns=patterns,
    )


if __name__ == "__main__":
    main()

# Example:
# python -m SynFlow.Data.frame_stanza \
#   --input-dir /home/volt/bach/Corpora/raw_sentences \
#   --output-dir /home/volt/bach/Corpora/raw_sentences_frame_parsed \
#   --frame-model base \
#   --small-frame-model small \
#   --language en \
#   --stanza-package ewt \
#   --gpu 2,3 \
#   --workers-per-gpu 1 \
#   --frame-batch-size 24
