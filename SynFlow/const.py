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
    "token/pos_init",
    "token/deprel",

    "lemma_only",
    "lemma/pos",
    "lemma/pos_init",
    "lemma/deprel",
}
