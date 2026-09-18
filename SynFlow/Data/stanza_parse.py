"""Parse raw sentence files with Stanza.

The input corpus is expected to contain raw sentence files inside subfolders
below an input root, where each line is one raw sentence.

    sentence_text

The output mirrors the input directory structure and writes one parsed sentence
block per non-empty input row. Sentence ids are generated from the input file's
parent directory name, file stem, and input line number in that file:

    <s id=PARENT_DIRECTORY_FILE_STEM_LINE_NUMBER>
    token<TAB>lemma<TAB>upos<TAB>id<TAB>head<TAB>deprel<TAB>feats
    </s>
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from tqdm import tqdm

if TYPE_CHECKING:
    import stanza


PIPELINE: stanza.Pipeline | None = None
PROGRESS_QUEUE: Queue[int] | None = None
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
    progress_queue: Queue[int],
    language: str,
    model: str | None,
    processors: str,
    processor_models: Mapping[str, str] | None,
) -> None:
    """Load one Stanza pipeline on the GPU assigned to this worker."""
    global PIPELINE, PROGRESS_QUEUE
    PROGRESS_QUEUE = progress_queue
    PIPELINE = make_pipeline(
        gpu=gpu_queue.get(),
        language=language,
        model=model,
        processors=processors,
        processor_models=processor_models,
    )


def report_file_progress() -> None:
    """Notify the parent process that one file has finished."""
    if PROGRESS_QUEUE is not None:
        PROGRESS_QUEUE.put(1)


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


def sentence_id_base(input_path: Path, sentence_index: str) -> str:
    """Build the shared SynFlow sentence id base for one input line."""
    return f"{input_path.parent.name}_{input_path.stem}_{sentence_index}"


def format_sentence_block(sentence_id: str, doc: object) -> str:
    """Serialize one Stanza document as one sentence block."""
    lines = [f"<s id={sentence_id}>"]
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
    return parse_task_group(
        [task],
        batch_size,
        overwrite,
        show_sentence_progress,
    )[0]


def parse_task_group(
    tasks: list[ParseTask],
    batch_size: int,
    overwrite: bool,
    show_sentence_progress: bool,
) -> list[ParseResult]:
    """Parse sentences from several files in shared Stanza batches."""
    if PIPELINE is None:
        raise RuntimeError("Stanza pipeline was not initialized")

    results: list[ParseResult] = []
    pending: list[tuple[ParseTask, str, str, bool]] = []
    sentence_total = 0

    for task in tasks:
        if task.output_path.exists() and not overwrite:
            results.append(
                ParseResult(task.input_path, task.output_path, 0, skipped=True)
            )
            report_file_progress()
            continue

        rows = read_sentence_rows(task.input_path)
        sentence_total += len(rows)
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = task.output_path.with_name(f".{task.output_path.name}.tmp")
        tmp_path.unlink(missing_ok=True)
        tmp_path.touch()

        if not rows:
            with tmp_path.open("a", encoding="utf-8") as output_file:
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(tmp_path, task.output_path)
            fsync_parent(task.output_path)
            results.append(
                ParseResult(task.input_path, task.output_path, 0, skipped=False)
            )
            report_file_progress()
            continue

        for row_index, (sentence_index, text) in enumerate(rows):
            pending.append((task, sentence_index, text, row_index == len(rows) - 1))
            if len(pending) == batch_size:
                _process_cross_file_batch(pending)
                pending.clear()

        results.append(
            ParseResult(task.input_path, task.output_path, len(rows), skipped=False)
        )

    if pending:
        _process_cross_file_batch(pending)

    if show_sentence_progress:
        tqdm.write(f"Parsed {sentence_total} sentences from {len(tasks)} files")
    return results


def _process_cross_file_batch(
    batch: list[tuple[ParseTask, str, str, bool]],
) -> None:
    """Parse one batch and route documents back to their source files."""
    if PIPELINE is None:
        raise RuntimeError("Stanza pipeline was not initialized")

    docs = PIPELINE.bulk_process([text for _, _, text, _ in batch])
    if len(docs) != len(batch):
        raise RuntimeError(
            f"Stanza returned {len(docs)} docs for {len(batch)} inputs"
        )

    blocks_by_task: dict[ParseTask, list[str]] = {}
    completed_tasks: set[ParseTask] = set()
    for (task, sentence_index, _, is_last), doc in zip(batch, docs):
        blocks_by_task.setdefault(task, []).append(
            format_sentence_block(
                sentence_id_base(task.input_path, sentence_index),
                doc,
            )
        )
        if is_last:
            completed_tasks.add(task)

    for task, blocks in blocks_by_task.items():
        tmp_path = task.output_path.with_name(f".{task.output_path.name}.tmp")
        with tmp_path.open("a", encoding="utf-8") as output_file:
            output_file.write("\n".join(blocks))
            output_file.write("\n")
            if task in completed_tasks:
                output_file.flush()
                os.fsync(output_file.fileno())

        if task in completed_tasks:
            os.replace(tmp_path, task.output_path)
            fsync_parent(task.output_path)
            report_file_progress()


def balance_tasks_by_size(
    tasks: list[ParseTask],
    worker_count: int,
) -> list[list[ParseTask]]:
    """Distribute files across workers while balancing their total byte sizes."""
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    if not tasks:
        return []

    group_count = min(worker_count, len(tasks))
    sized_tasks = sorted(
        ((task.input_path.stat().st_size, task) for task in tasks),
        key=lambda item: item[0],
        reverse=True,
    )
    groups: list[list[ParseTask]] = [[] for _ in range(group_count)]
    group_sizes: list[tuple[int, int, int]] = []

    for group_index, (file_size, task) in enumerate(sized_tasks[:group_count]):
        groups[group_index].append(task)
        group_sizes.append((file_size, 1, group_index))
    heapq.heapify(group_sizes)

    for file_size, task in sized_tasks[group_count:]:
        total_size, file_count, group_index = heapq.heappop(group_sizes)
        groups[group_index].append(task)
        heapq.heappush(
            group_sizes,
            (total_size + file_size, file_count + 1, group_index),
        )

    return groups


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
    pending_tasks = (
        tasks
        if overwrite
        else [task for task in tasks if not task.output_path.exists()]
    )
    skipped_files = len(tasks) - len(pending_tasks)

    if not pending_tasks:
        with tqdm(
            total=len(tasks),
            initial=skipped_files,
            desc="Files",
            unit="file",
        ):
            pass
        print(
            f"Done. Files: {len(tasks)}, skipped: {skipped_files}, "
            "sentences parsed: 0",
            flush=True,
        )
        return

    gpu_queue: Queue[int] = Queue()
    progress_queue: Queue[int] = Queue()
    show_sentence_progress = len(worker_gpu_ids) == 1
    for gpu_id in worker_gpu_ids:
        gpu_queue.put(gpu_id)

    with ProcessPoolExecutor(
        max_workers=len(worker_gpu_ids),
        initializer=init_worker,
        initargs=(
            gpu_queue,
            progress_queue,
            language,
            model,
            processors,
            processor_models,
        ),
    ) as executor:
        task_groups = balance_tasks_by_size(pending_tasks, len(worker_gpu_ids))
        futures = [
            executor.submit(
                parse_task_group,
                task_group,
                batch_size,
                overwrite,
                show_sentence_progress,
            )
            for task_group in task_groups
        ]
        pending_futures = set(futures)
        with tqdm(
            total=len(tasks),
            initial=skipped_files,
            desc="Files",
            unit="file",
        ) as file_progress:
            while pending_futures:
                try:
                    file_progress.update(progress_queue.get(timeout=0.2))
                except Empty:
                    pass

                completed_futures = {
                    future for future in pending_futures if future.done()
                }
                for future in completed_futures:
                    results = future.result()
                    for result in results:
                        parsed_sentences += result.sentence_count
                        skipped_files += int(result.skipped)
                    pending_futures.remove(future)

            while file_progress.n < len(tasks):
                try:
                    file_progress.update(progress_queue.get_nowait())
                except Empty:
                    file_progress.update(len(tasks) - file_progress.n)

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
