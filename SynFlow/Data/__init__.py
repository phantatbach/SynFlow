"""Data preparation utilities for SynFlow workflows."""

from .stanza_parse import (
    DEFAULT_FILE_EXTENSIONS,
    DEFAULT_PROCESSORS,
    ParseResult,
    ParseTask,
    stanza_parse_folder,
)
from .w2v_data import (
    COLUMN_INDEX,
    merge_txt_files_all_subfolders,
    merge_txt_files_one_subfolder,
    parsed_file_to_sentences,
    parsed_file_to_sentences_folder,
    write_selected_column_text,
)

__all__ = [
    "COLUMN_INDEX",
    "DEFAULT_FILE_EXTENSIONS",
    "DEFAULT_PROCESSORS",
    "ParseResult",
    "ParseTask",
    "frame_stanza_parse_folder",
    "hanlp_stanza_parse_folder",
    "merge_txt_files_all_subfolders",
    "merge_txt_files_one_subfolder",
    "parsed_file_to_sentences",
    "parsed_file_to_sentences_folder",
    "stanza_parse_folder",
    "write_selected_column_text",
]


def __getattr__(name: str) -> object:
    """Lazily expose optional parser entry points."""
    if name == "frame_stanza_parse_folder":
        from .frame_stanza import frame_stanza_parse_folder

        return frame_stanza_parse_folder
    if name == "hanlp_stanza_parse_folder":
        from .hanlp_stanza_srl import hanlp_stanza_parse_folder

        return hanlp_stanza_parse_folder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
