"""Parse raw sentence files with Stanza.

The input corpus is expected to contain raw sentence files inside subfolders
below an input root, where each line is one raw sentence.

    sentence_text

The output mirrors the input directory structure and writes one parsed sentence
block per non-empty input row. Sentence ids are generated from the input file's
parent directory name and the input line number in that file:

    <s id=PARENT_DIRECTORY_LINE_NUMBER>
    token<TAB>lemma<TAB>upos<TAB>id<TAB>head<TAB>deprel<TAB>feats
    </s>
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from tqdm import tqdm

if TYPE_CHECKING:
    import stanza


PIPELINE: stanza.Pipeline | None = None
DEFAULT_FILE_EXTENSIONS = ("*.txt", "*.conll", "*.conllu", "*.json")
DEFAULT_PROCESSORS = "tokenize,mwt,pos,lemma,depparse"


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
        description="Parse raw sentence files into a Stanza dependency format."
    )
    parser.add_argument(
        "--input-root",
        "--input-dir",
        dest="input_root",
        type=Path,
        default=Path.cwd(),
        help="Corpus root containing raw sentence files. Default: current directory.",
    )
    parser.add_argument(
        "--output-root",
        "--output-dir",
        dest="output_root",
        type=Path,
        default=None,
        help=(
            "Output root. Default: sibling directory named "
            "<input-root-name>_parsed."
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
        "--batch-size",
        type=int,
        default=128,
        help="Number of input rows parsed in each Stanza batch.",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Required Stanza language code, for example de or en.",
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        help="One Stanza package/model to use for all processors, for example gsd or ewt.",
    )
    model_group.add_argument(
        "--processor-models-json",
        help=(
            "JSON object mapping each processor to a package/model, for example "
            '\'{"tokenize":"gsd","pos":"hdt","lemma":"hdt","depparse":"hdt"}\'.'
        ),
    )
    parser.add_argument(
        "--processors",
        default=DEFAULT_PROCESSORS,
        help=(
            "Comma-separated Stanza processors to run when --model is used. "
            f"Default: {DEFAULT_PROCESSORS}."
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
        raise ValueError("--workers-per-gpu must be at least 1")
    return [gpu_id for gpu_id in gpu_ids for _ in range(workers_per_gpu)]


def parse_processor_models_json(raw_json: str | None) -> dict[str, str] | None:
    """Parse CLI JSON for Stanza per-processor model packages."""
    if raw_json is None:
        return None

    processor_models = json.loads(raw_json)
    if not isinstance(processor_models, dict):
        raise ValueError("--processor-models-json must be a JSON object")

    for processor, model in processor_models.items():
        if not isinstance(processor, str) or not isinstance(model, str):
            raise ValueError("--processor-models-json keys and values must be strings")

    return processor_models


def resolve_processor_models(
    processor_models: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Return a plain dict for per-processor model config."""
    return dict(processor_models) if processor_models is not None else None


def validate_pipeline_config(
    language: str,
    model: str | None,
    processor_models: Mapping[str, str] | None,
) -> str:
    """Validate required Stanza language and model configuration."""
    language = language.strip()
    if not language:
        raise ValueError("language must be a non-empty Stanza language code")

    if model is None and processor_models is None:
        raise ValueError("Either model or processor_models must be provided")
    if model is not None and processor_models is not None:
        raise ValueError("Use either model or processor_models, not both")
    if model is not None and not model.strip():
        raise ValueError("model must be a non-empty Stanza package name")
    if processor_models is not None and not processor_models:
        raise ValueError("processor_models must not be empty")

    return language


def make_pipeline(
    gpu: int,
    language: str,
    model: str | None,
    processors: str,
    processor_models: Mapping[str, str] | None,
) -> stanza.Pipeline:
    """Load one Stanza pipeline on one CUDA GPU."""
    import stanza
    from stanza.pipeline.core import DownloadMethod

    device = f"cuda:{gpu}"
    resolved_processor_models = resolve_processor_models(processor_models)
    resolved_package = None if resolved_processor_models is not None else model
    resolved_processors: str | dict[str, str] = (
        resolved_processor_models
        if resolved_processor_models is not None
        else processors
    )

    print(
        f"Loading Stanza {language} on {device} "
        f"with package={resolved_package!r}, processors={resolved_processors!r}",
        flush=True,
    )

    return stanza.Pipeline(
        lang=language,
        package=resolved_package,
        processors=resolved_processors,
        tokenize_no_ssplit=True,
        use_gpu=True,
        device=device,
        download_method=DownloadMethod.REUSE_RESOURCES,
    )


def init_worker(
    gpu_queue: Queue[int],
    language: str,
    model: str | None,
    processors: str,
    processor_models: Mapping[str, str] | None,
) -> None:
    """Load one Stanza pipeline on the GPU assigned to this worker."""
    global PIPELINE
    PIPELINE = make_pipeline(
        gpu=gpu_queue.get(),
        language=language,
        model=model,
        processors=processors,
        processor_models=processor_models,
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


def stanza_parse_folder(
    input_root: str | Path,
    output_root: str | Path,
    *,
    language: str,
    model: str | None,
    processors: str = DEFAULT_PROCESSORS,
    processor_models: Mapping[str, str] | None = None,
    gpu: str = "0",
    workers_per_gpu: int = 1,
    batch_size: int = 128,
    overwrite: bool = False,
    file_patterns: Sequence[str] = DEFAULT_FILE_EXTENSIONS,
) -> Path:
    """Parse raw sentence files inside input-root subfolders."""
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    language = validate_pipeline_config(language, model, processor_models)
    effective_processor_models = processor_models

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
        f"Language: {language}; model: {model}; "
        f"processors: "
        f"{effective_processor_models if effective_processor_models is not None else processors}",
        flush=True,
    )
    print(
        f"GPUs: {', '.join(f'cuda:{gpu_id}' for gpu_id in gpu_ids)}; "
        f"workers per GPU: {workers_per_gpu}; "
        f"total workers: {len(worker_gpu_ids)}; "
        f"batch size: {batch_size}; "
        f"overwrite: {overwrite}",
        flush=True,
    )

    run_tasks(
        tasks=tasks,
        batch_size=batch_size,
        worker_gpu_ids=worker_gpu_ids,
        overwrite=overwrite,
        language=language,
        model=model,
        processors=processors,
        processor_models=effective_processor_models,
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
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            rows.append((str(line_number), stripped))
    return rows


def format_sentence_block(parent_name: str, sentence_index: str, doc: object) -> str:
    """Serialize one Stanza document as one sentence block."""
    lines = [f"<s id={parent_name}_{sentence_index}>"]
    sentences = getattr(doc, "sentences")
    for sentence in sentences:
        for word in sentence.words:
            feats = word.feats or "-"
            lines.append(
                "\t".join(
                    [
                        word.text,
                        word.lemma or "_",
                        word.upos or "_",
                        str(word.id),
                        str(word.head),
                        word.deprel or "_",
                        feats,
                    ]
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
    batch_size: int,
    overwrite: bool,
    show_sentence_progress: bool,
) -> ParseResult:
    """Parse one corpus file and atomically publish the mirrored output."""
    if task.output_path.exists() and not overwrite:
        return ParseResult(task.input_path, task.output_path, 0, skipped=True)
    if PIPELINE is None:
        raise RuntimeError("Stanza pipeline was not initialized")

    rows = read_sentence_rows(task.input_path)
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = task.output_path.with_name(f".{task.output_path.name}.tmp")
    tmp_path.unlink(missing_ok=True)

    with (
        tmp_path.open("w", encoding="utf-8") as output_file,
        tqdm(
            total=len(rows),
            desc=task.input_path.name,
            unit="sent",
            leave=False,
            disable=not show_sentence_progress,
        ) as sentence_progress,
    ):
        for batch in batched(rows, batch_size):
            indexes = [sentence_index for sentence_index, _ in batch]
            texts = [sentence for _, sentence in batch]
            docs = PIPELINE.bulk_process(texts)

            if len(docs) != len(indexes):
                raise RuntimeError(
                    f"Stanza returned {len(docs)} docs for {len(indexes)} inputs"
                )

            for sentence_index, doc in zip(indexes, docs):
                output_file.write(
                    format_sentence_block(
                        task.input_path.parent.name,
                        sentence_index,
                        doc,
                    )
                )
                output_file.write("\n")

            sentence_progress.update(len(batch))

        output_file.flush()
        os.fsync(output_file.fileno())

    os.replace(tmp_path, task.output_path)
    fsync_parent(task.output_path)
    return ParseResult(task.input_path, task.output_path, len(rows), skipped=False)


def run_tasks(
    tasks: list[ParseTask],
    batch_size: int,
    worker_gpu_ids: list[int],
    overwrite: bool,
    language: str,
    model: str | None,
    processors: str,
    processor_models: Mapping[str, str] | None,
) -> None:
    """Parse files in parallel with the requested GPU worker layout."""
    parsed_sentences = 0
    skipped_files = 0
    gpu_queue: Queue[int] = Queue()
    show_sentence_progress = len(worker_gpu_ids) == 1
    for gpu_id in worker_gpu_ids:
        gpu_queue.put(gpu_id)

    with ProcessPoolExecutor(
        max_workers=len(worker_gpu_ids),
        initializer=init_worker,
        initargs=(gpu_queue, language, model, processors, processor_models),
    ) as executor:
        futures = [
            executor.submit(
                parse_file,
                task,
                batch_size,
                overwrite,
                show_sentence_progress,
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
        else input_root.parent / f"{input_root.name}_parsed"
    )

    patterns = args.pattern if args.pattern is not None else DEFAULT_FILE_EXTENSIONS
    processor_models = parse_processor_models_json(args.processor_models_json)
    stanza_parse_folder(
        input_root=input_root,
        output_root=output_root,
        language=args.language,
        model=args.model,
        processors=args.processors,
        processor_models=processor_models,
        gpu=args.gpu,
        workers_per_gpu=args.workers_per_gpu,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        file_patterns=patterns,
    )


if __name__ == "__main__":
    main()

# Example:
# python -m SynFlow.Data.stanza_parse \
#   --input-dir /home/volt/bach/Corpora/raw_sentences \
#   --output-dir /home/volt/bach/Corpora/raw_sentences_parsed \
#   --language en \
#   --model ewt \
#   --gpu 2,3 \
#   --workers-per-gpu 2 \
#   --batch-size 256
