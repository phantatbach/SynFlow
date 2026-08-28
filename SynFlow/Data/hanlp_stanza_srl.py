"""Parse raw sentence files with HanLP SRL and Stanza lemmatisation.

The input corpus is expected to contain raw sentence files inside subfolders
below an input root, where each line is one raw sentence.

The output mirrors the input directory structure and writes one SRL frame block
per predicate:

    <id=FILE_STEM_LINE_NUMBER_PREDICATE_NUMBER>
    srl_component<TAB>lemmatised_srl_component<TAB>-<TAB>component_id<TAB>head_id<TAB>
    srl_relation<TAB>-
    <s>
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm import tqdm


SRL_PIPELINE: Any | None = None
LEMMA_PIPELINE: Any | None = None
DEFAULT_FILE_EXTENSIONS = ("*.txt", "*.conll", "*.conllu", "*.json")
DEFAULT_STANZA_PROCESSORS = "tokenize,pos,lemma"
DEFAULT_HANLP_BATCH_SIZE = 32


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
    skipped: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse raw sentence files into HanLP SRL plus Stanza lemma "
            "SynFlow format."
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
            "<input-root-name>_srl_parsed."
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
        "--hanlp-model",
        required=True,
        help="Required HanLP SRL model name, URL, or local path.",
    )
    parser.add_argument(
        "--hanlp-batch-size",
        type=int,
        default=DEFAULT_HANLP_BATCH_SIZE,
        help=(
            "Number of raw sentences sent to HanLP in each batch. "
            f"Default: {DEFAULT_HANLP_BATCH_SIZE}."
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


def validate_parse_config(
    hanlp_model_name: str,
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str,
    hanlp_batch_size: int,
) -> tuple[str, str]:
    """Validate required HanLP and Stanza parser configuration."""
    hanlp_model_name = hanlp_model_name.strip()
    language = language.strip()
    stanza_processors = stanza_processors.strip()

    if not hanlp_model_name:
        raise ValueError("hanlp_model_name must be a non-empty HanLP model name")
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
    if hanlp_batch_size < 1:
        raise ValueError("hanlp_batch_size must be at least 1")

    return hanlp_model_name, language


def build_hanlp_srl_pipeline(model_name: str, gpu: int) -> Any:
    """Load one HanLP SRL pipeline on one CUDA GPU."""
    import hanlp

    device = f"cuda:{gpu}"
    print(f"Loading HanLP SRL on {device} with model={model_name!r}", flush=True)
    return hanlp.load(model_name, devices=gpu)


def build_stanza_lemma_pipeline(
    gpu: int,
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str,
) -> Any:
    """Load one Stanza lemmatisation pipeline on one CUDA GPU."""
    import stanza
    from stanza.pipeline.core import DownloadMethod

    device = f"cuda:{gpu}"
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
        use_gpu=True,
        device=device,
        download_method=DownloadMethod.REUSE_RESOURCES,
    )


def init_worker(
    gpu_queue: Queue[int],
    hanlp_model_name: str,
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str,
) -> None:
    """Load HanLP and Stanza pipelines on the GPU assigned to this worker."""
    global SRL_PIPELINE, LEMMA_PIPELINE
    gpu = gpu_queue.get()
    SRL_PIPELINE = build_hanlp_srl_pipeline(hanlp_model_name, gpu)
    LEMMA_PIPELINE = build_stanza_lemma_pipeline(
        gpu=gpu,
        language=language,
        stanza_package=stanza_package,
        stanza_processors=stanza_processors,
    )


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


def hanlp_stanza_parse_folder(
    input_root: str | Path,
    output_root: str | Path,
    *,
    hanlp_model_name: str,
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str = DEFAULT_STANZA_PROCESSORS,
    gpu: str = "0",
    workers_per_gpu: int = 1,
    hanlp_batch_size: int = DEFAULT_HANLP_BATCH_SIZE,
    overwrite: bool = False,
    file_patterns: Sequence[str] = DEFAULT_FILE_EXTENSIONS,
) -> Path:
    """Parse raw sentence files inside input-root subfolders."""
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    hanlp_model_name, language = validate_parse_config(
        hanlp_model_name=hanlp_model_name,
        language=language,
        stanza_package=stanza_package,
        stanza_processors=stanza_processors,
        hanlp_batch_size=hanlp_batch_size,
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
        f"HanLP model: {hanlp_model_name}; "
        f"Stanza language: {language}; "
        f"Stanza package: {stanza_package}; "
        f"Stanza processors: {stanza_processors}",
        flush=True,
    )
    print(
        f"GPUs: {', '.join(f'cuda:{gpu_id}' for gpu_id in gpu_ids)}; "
        f"workers per GPU: {workers_per_gpu}; "
        f"total workers: {len(worker_gpu_ids)}; "
        f"HanLP batch size: {hanlp_batch_size}; "
        f"overwrite: {overwrite}",
        flush=True,
    )

    run_tasks(
        tasks=tasks,
        hanlp_batch_size=hanlp_batch_size,
        worker_gpu_ids=worker_gpu_ids,
        overwrite=overwrite,
        hanlp_model_name=hanlp_model_name,
        language=language,
        stanza_package=stanza_package,
        stanza_processors=stanza_processors,
    )
    return output_root


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
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


def parse_hanlp_srl_batches(
    sentences: list[str],
    hanlp_batch_size: int,
) -> list[Any]:
    """Run HanLP SRL in sentence batches and return SRL frames per sentence."""
    if SRL_PIPELINE is None:
        raise RuntimeError("HanLP SRL pipeline was not initialized")
    if not sentences:
        return []

    all_srl_frames: list[Any] = []
    for batch in batched(sentences, hanlp_batch_size):
        all_srl_frames.extend(SRL_PIPELINE(batch)["srl"])
    return all_srl_frames


def span_start(span: Any, fallback_index: int) -> int:
    """Return HanLP span start offset when available."""
    if len(span) >= 3 and isinstance(span[2], int):
        return span[2]
    return fallback_index


def lemmatise_component(srl_component: str) -> str:
    """Lemmatise one SRL component by parsing it with Stanza."""
    if LEMMA_PIPELINE is None:
        raise RuntimeError("Stanza lemma pipeline was not initialized")

    doc = LEMMA_PIPELINE(srl_component)
    lemmas = [
        word.lemma or word.text
        for sentence in doc.sentences
        for word in sentence.words
    ]
    return " ".join(lemmas) if lemmas else srl_component


def frame_to_rows(frame: list[Any]) -> list[tuple[str, str, int, int, str]]:
    """Convert one HanLP SRL frame into output rows."""
    sorted_frame = sorted(
        enumerate(frame),
        key=lambda item: span_start(item[1], item[0]),
    )
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
        lemmatised_srl_component = lemmatise_component(srl_component)
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


def format_frame_block(
    file_id: str,
    sentence_index: str,
    predicate_index: int,
    frame: list[Any],
) -> str | None:
    """Serialize one HanLP SRL frame as one SynFlow block."""
    rows = frame_to_rows(frame)
    if not rows:
        return None

    lines = [f"<id={file_id}_{sentence_index}_{predicate_index}>"]
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
    hanlp_batch_size: int,
    overwrite: bool,
) -> ParseResult:
    """Parse one corpus file and atomically publish the mirrored output."""
    if task.output_path.exists() and not overwrite:
        return ParseResult(task.input_path, task.output_path, 0, skipped=True)

    rows = read_sentence_rows(task.input_path)
    sentence_indexes = [sentence_index for sentence_index, _ in rows]
    sentences = [sentence for _, sentence in rows]
    all_srl_frames = parse_hanlp_srl_batches(sentences, hanlp_batch_size)

    if len(all_srl_frames) != len(sentence_indexes):
        raise RuntimeError(
            f"HanLP returned {len(all_srl_frames)} SRL results "
            f"for {len(sentence_indexes)} inputs"
        )

    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = task.output_path.with_name(f".{task.output_path.name}.tmp")
    tmp_path.unlink(missing_ok=True)

    with tmp_path.open("w", encoding="utf-8") as output_file:
        for sentence_index, sentence_frames in zip(
            sentence_indexes,
            all_srl_frames,
        ):
            predicate_index = 0
            for frame in sentence_frames:
                block = format_frame_block(
                    file_id=task.input_path.stem,
                    sentence_index=sentence_index,
                    predicate_index=predicate_index + 1,
                    frame=frame,
                )
                if block is None:
                    continue
                predicate_index += 1
                output_file.write(block)
                output_file.write("\n")

        output_file.flush()
        os.fsync(output_file.fileno())

    os.replace(tmp_path, task.output_path)
    fsync_parent(task.output_path)
    return ParseResult(task.input_path, task.output_path, len(rows), skipped=False)


def run_tasks(
    tasks: list[ParseTask],
    hanlp_batch_size: int,
    worker_gpu_ids: list[int],
    overwrite: bool,
    hanlp_model_name: str,
    language: str,
    stanza_package: str | Mapping[str, str],
    stanza_processors: str,
) -> None:
    """Parse files in parallel with the requested GPU worker layout."""
    parsed_sentences = 0
    skipped_files = 0
    gpu_queue: Queue[int] = Queue()
    for gpu_id in worker_gpu_ids:
        gpu_queue.put(gpu_id)

    with ProcessPoolExecutor(
        max_workers=len(worker_gpu_ids),
        initializer=init_worker,
        initargs=(
            gpu_queue,
            hanlp_model_name,
            language,
            stanza_package,
            stanza_processors,
        ),
    ) as executor:
        futures = [
            executor.submit(
                parse_file,
                task,
                hanlp_batch_size,
                overwrite,
            )
            for task in tasks
        ]
        with tqdm(total=len(futures), desc="Files", unit="file") as file_progress:
            for future in as_completed(futures):
                result = future.result()
                parsed_sentences += result.sentence_count
                skipped_files += int(result.skipped)
                if result.skipped:
                    file_progress.set_postfix_str(
                        f"skipped: {result.input_path.name}"
                    )
                else:
                    file_progress.set_postfix_str(
                        f"parsed {result.sentence_count}: "
                        f"{result.input_path.name}"
                    )
                file_progress.update(1)

    print(
        f"Done. Files: {len(tasks)}, skipped: {skipped_files}, "
        f"sentences parsed: {parsed_sentences}",
        flush=True,
    )


def main() -> None:
    """Parse all discovered corpus files."""
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else input_root.parent / f"{input_root.name}_srl_parsed"
    )

    patterns = args.pattern if args.pattern is not None else DEFAULT_FILE_EXTENSIONS
    stanza_package = (
        parse_stanza_package_json(args.stanza_package_json)
        if args.stanza_package_json is not None
        else args.stanza_package
    )
    if stanza_package is None:
        raise ValueError("A Stanza package must be provided")

    hanlp_stanza_parse_folder(
        input_root=input_root,
        output_root=output_root,
        hanlp_model_name=args.hanlp_model,
        language=args.language,
        stanza_package=stanza_package,
        stanza_processors=args.stanza_processors,
        gpu=args.gpu,
        workers_per_gpu=args.workers_per_gpu,
        hanlp_batch_size=args.hanlp_batch_size,
        overwrite=args.overwrite,
        file_patterns=patterns,
    )


if __name__ == "__main__":
    main()

# Example:
# python -m SynFlow.Data.hanlp_stanza_srl \
#   --input-dir /home/volt/bach/Corpora/raw_sentences \
#   --output-dir /home/volt/bach/Corpora/raw_sentences_srl_parsed \
#   --hanlp-model /path/to/hanlp-srl-model \
#   --language en \
#   --stanza-package-json '{"tokenize":"ewt","pos":"ewt","lemma":"ewt"}' \
#   --gpu 2,3 \
#   --workers-per-gpu 1 \
#   --hanlp-batch-size 32
