import re

TARGET_ALL_POS = "ALLPOS"

DEFAULT_PATTERN = re.compile(
    r"([^\t]+)\t"      # word form
    r"([^\t]+)\t"      # lemma
    r"([^\t]+)\t"      # POS
    r"([^\t]+)\t"      # ID
    r"([^\t]+)\t"      # HEAD
    r"([^\t]+)\t"      # DEPREL
    r"([^\t]+)"        # FEATS
)

DEFAULT_COLS = ["id", "subfolder", "target"]

VALID_FILLER_FORMATS = {
    "token_only",
    "token/pos",
    "token/deprel",

    "lemma_only",
    "lemma/pos",
    "lemma/deprel",
}

SENT_ID_PATTERN = re.compile(r"<s\s+id=([^>]+)>")


def is_all_pos_target(target_pos: str) -> bool:
    """Return whether the target POS sentinel should match every POS tag."""
    return target_pos == TARGET_ALL_POS


def target_label(target_lemma: str, target_pos: str) -> str:
    """Return the public target label stored in output tables."""
    return f"{target_lemma}/{target_pos}"


def target_line_contains(line: str, target_lemma: str, target_pos: str) -> bool:
    """Return whether a parsed token line can contain the requested target."""
    if is_all_pos_target(target_pos):
        return f"\t{target_lemma}\t" in line
    return f"\t{target_lemma}\t{target_pos}\t" in line


def target_matches(
    lemma: str,
    pos: str,
    target_lemma: str,
    target_pos: str,
) -> bool:
    """Return whether one token matches the requested lemma/POS target."""
    return lemma == target_lemma and (
        is_all_pos_target(target_pos) or pos == target_pos
    )


def lemma_pos_matches(
    lemma_pos: str,
    target_lemma: str,
    target_pos: str,
) -> bool:
    """Return whether a ``lemma/POS`` graph value matches the requested target."""
    lemma, pos = lemma_pos.rsplit("/", 1)
    return target_matches(lemma, pos, target_lemma, target_pos)
