from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from tqdm import tqdm

COLUMN_INDEX = {
    "token": 0,
    "lemma": 1,
    "pos": 2,
    "id": 3,
    "headid": 4,
    "deprel": 5,
    "feats": 6,
}

#----------------------------------------------
# Converting parsed files to whitespace-tokenized/lemmatised sentences
#----------------------------------------------

def parsed_file_to_sentences(
    input_path: str | Path,
    column: str = "lemma",
    drop_punct: bool = False,
) -> list[list[str]]:
    input_path = Path(input_path)
    column = column.lower()

    if column not in COLUMN_INDEX:
        valid_columns = ", ".join(sorted(COLUMN_INDEX))
        raise ValueError(
            f"Unknown column: {column}. Valid columns: {valid_columns}"
        )

    col_idx = COLUMN_INDEX[column]
    sentences = []
    current_sentence = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                continue

            if line.startswith("<s "):
                current_sentence = []
                continue

            if line == "</s>":
                if current_sentence:
                    sentences.append(current_sentence)

                current_sentence = []
                continue

            parts = line.split("\t")

            if len(parts) != 7:
                continue

            if drop_punct and parts[2] == "PUNCT":
                continue

            current_sentence.append(parts[col_idx])

    if current_sentence:
        sentences.append(current_sentence)

    return sentences

def write_selected_column_text(
    input_path: str | Path,
    output_path: str | Path,
    column: str = "lemma",
    drop_punct: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentences = parsed_file_to_sentences(
        input_path=input_path,
        column=column,
        drop_punct=drop_punct,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(" ".join(sentence) + "\n")

def parsed_file_to_sentences_1_file(
    input_path: Path,
    input_folder: Path,
    output_folder: Path,
    column: str,
    drop_punct: bool,
):
    relative_path = input_path.relative_to(input_folder)
    output_path = output_folder / relative_path

    write_selected_column_text(
        input_path=input_path,
        output_path=output_path,
        column=column,
        drop_punct=drop_punct,
    )

    return input_path

def parsed_file_to_sentences_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    column: str = "lemma",
    drop_punct: bool = False,
    workers: int | None = None,
) -> None:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.is_dir():
        raise NotADirectoryError(input_folder)

    input_files = sorted(input_folder.rglob("*.txt"))

    if workers is None:
        workers = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=workers) as executor:

        futures = [
            executor.submit(
                parsed_file_to_sentences_1_file,
                input_path,
                input_folder,
                output_folder,
                column,
                drop_punct,
            )
            for input_path in input_files
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing files",
        ):
            # Raises exception here if a worker failed
            future.result()
            
def merge_txt_files_one_subfolder(
    folder: Path,
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.txt",
) -> tuple[str, int, str]:
    """
    Merge all matching files directly inside one subfolder.

    Output filename = subfolder name.

    Returns:
        (folder_name, number_of_files, output_path)
    """

    files = sorted(
        path
        for path in folder.glob(pattern)
        if path.is_file()
    )

    if not files:
        return folder.name, 0, ""

    # Preserve relative folder structure
    relative_folder = folder.relative_to(input_dir)

    # Output filename = subfolder name
    output_filename = f"{folder.name}.txt"

    output_path = (
        output_dir
        / relative_folder
        / output_filename
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as out_f:

        for file_path in files:
            last_char_was_newline = True

            with file_path.open(
                "r",
                encoding="utf-8",
            ) as in_f:

                for line in in_f:
                    out_f.write(line)
                    last_char_was_newline = line.endswith("\n")

            # Ensure next file starts on a new line
            if not last_char_was_newline:
                out_f.write("\n")

    return (
        folder.name,
        len(files),
        str(output_path),
    )

def merge_txt_files_all_subfolders(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.txt",
    workers: int | None = None,
) -> None:
    """
    Merge txt files by subfolder using multiprocessing.

    Each subfolder is one task.
    Multiple subfolders can be processed in parallel.
    """

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {input_dir}"
        )

    # Find folders that directly contain matching files
    folders = sorted({
        file_path.parent
        for file_path in input_dir.rglob(pattern)
        if file_path.is_file()
    })

    if not folders:
        print("No matching files found.")
        return

    if workers is None:
        workers = max(
            1,
            (os.cpu_count() or 1) - 1,
        )

    print(f"Found {len(folders)} subfolders")
    print(f"Using {workers} processes")

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                merge_txt_files_one_subfolder,
                folder,
                input_dir,
                output_dir,
                pattern,
            ): folder
            for folder in folders
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Merging folders",
        ):
            folder = futures[future]

            try:
                folder_name, num_files, output_path = future.result()

                if num_files > 0:
                    tqdm.write(
                        f"{folder_name}: "
                        f"{num_files} files -> {output_path}"
                    )

            except Exception as e:
                tqdm.write(
                    f"ERROR: {folder} -> {e}"
                )
