import re

DEFAULT_PATTERN = re.compile(
    r'([^\t]+)\t'      # word form
    r'([^\t]+)\t'      # lemma
    r'([^\t]+)\t'      # POS
    r'([^\t]+)\t'      # ID
    r'([^\t]+)\t'      # HEAD
    r'([^\t]+)'        # DEPREL
)

DEFAULT_COLS = ['id', 'subfolder', 'target']

VALID_FILLER_FORMATS = {
    "token_only",
    "token/pos",
    "token/deprel",

    "lemma_only",
    "lemma/pos",
    "lemma/deprel",
}

SENT_ID_PATTERN = re.compile(r"<s\s+id=([^>]+)>")