import re
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import pandas as pd

from SynFlow.utils import build_graph, format_filler
from SynFlow.const import DEFAULT_PATTERN, VALID_FILLER_FORMATS

def build_context_lookup(
    sent_tokens: List[str],
    pattern: re.Pattern,
) -> Dict[str, tuple[str, str, str]]:
    """Build a token-id to token, lemma, and POS lookup for one sentence."""
    id2context = {}
    for line in sent_tokens:
        m = pattern.match(line)
        if not m:
            continue
        token, lemma, pos, idx, _, _ = m.groups()
        id2context[idx] = (token, lemma, pos)
    return id2context

def find_by_path(graph, id2context, id2deprel, tgt_ids, rel, filler_format):
    """
    Finds all paths in a dependency graph starting from a set of target IDs that match
    a specified sequence of dependency relations.

    Args:
        graph (dict): Adjacency list representation of the dependency graph.
                      Keys are parent IDs, values are lists of child IDs.
        id2context (dict): Maps token ID to its token, lemma, and POS fields.
        id2deprel (dict): Maps (parent_id, child_id) tuple to dependency relation string.
        tgt_ids (list): A list of target token IDs.
        rel (str): A single relation path string, e.g., "chi_obl > chi_case".
                   '>' separates sequential steps.

    Returns:
        list: A list of tuples, where each tuple contains:
              - Formatted context words joined by " > ".
              - The actual path string found (e.g., "chi_obl > chi_case").
              Returns an empty list if no path is found.
    """
    seq = [r.strip() for r in rel.split('>')]
    N   = len(seq)
    out = []

    def dfs(node, depth, seen, path_rels, path_nodes):
        if depth == N:
            out.append(( " > ".join(path_nodes),
                         " > ".join(path_rels) ))
            return
        expected_rel = seq[depth]
        for nb in graph[node]:
            if nb in seen:                # ← block revisits, including target
                continue
            lbl = id2deprel.get((node, nb))
            if lbl == expected_rel:
                token, lemma, pos = id2context[nb]
                filler = format_filler(token, lemma, pos, lbl, filler_format)
                dfs(nb,
                    depth+1,
                    seen | {nb},
                    path_rels + [lbl],
                    path_nodes + [filler])

    for t in tgt_ids:
        dfs(t, 0, {t}, [], [])

    return out

def process_file(
        args: Tuple[str, str, Optional[re.Pattern], str, str, str, str]
        ) -> List[dict]:
    """
    args = (corpus_folder, fname, pattern, target_lemma, target_pos, rel, filler_format)
    returns one dict per matched path.
    """
    corpus_folder, fname, pattern, target_lemma, target_pos, rel, filler_format = args
    pattern = pattern or DEFAULT_PATTERN

    has_target = False
    has_target_check_string = f'\t{target_lemma}\t{target_pos}'

    results = []
    filepath = os.path.join(corpus_folder, fname)
    with open(filepath, encoding='utf8') as fh:
        sent_tokens, sent_forms = [], [] # Init for the whole file. Sent_tokens = lines, sent_forms = word forms only

        for line in fh:
            line = line.rstrip("\n")

            # Start a new sentence
            if line.startswith("<s id"):
                has_target = False # Reset for new sentence
                sent_tokens, sent_forms = [], [] # Reset for new sentence

            # End of a sentence. Build graph and process if target found
            elif line.startswith("</s>"):
                # Process the sentence if it contains the target lemma/POS
                if sent_tokens and has_target == True:
                    id2wp, graph, id2d = build_graph(sent_tokens, pattern)
                    id2context = build_context_lookup(sent_tokens, pattern)
                    sentence_text = " ".join(sent_forms)
                    target_lp = f"{target_lemma}/{target_pos}"
                    tgt_ids = [tid for tid, lp in id2wp.items() if lp == target_lp]
                    for sfillers, path_str in find_by_path(graph, id2context, id2d, tgt_ids, rel, filler_format):
                        results.append({
                            "file": fname,
                            "sentence": sentence_text,
                            "sfillers": sfillers,
                            "path": path_str,
                        })

            else:
                sent_tokens.append(line)
                m = pattern.match(line)
                if m:
                    sent_forms.append(m.group(1))
                
                if has_target_check_string in line:
                    has_target = True

    return results

def rel_explorer(corpus_folder: str,
                 pattern: re.Pattern = None,
                 target_lemma: str = None,
                 target_pos: str   = None,
                 deprel: str          = None,
                 filler_format: str = "lemma/pos",
                 num_processes: int= max(1, cpu_count() - 1)
                ) -> pd.DataFrame:
    """
    Walk corpus_folder in parallel and return matched relation paths as a DataFrame.

    Args:
        filler_format: Format for context words in the ``sfillers`` column.
            Must be one of ``"lemma_only"``, ``"lemma/pos"``,
            ``"lemma/pos_init"``, ``"lemma/deprel"``, ``"token_only"``,
            ``"token/pos"``, ``"token/pos_init"``, or ``"token/deprel"``.
    """
    pattern       = pattern or DEFAULT_PATTERN
    num_procs     = num_processes or max(1, cpu_count()-1)
    if filler_format not in VALID_FILLER_FORMATS:
        valid_formats = ", ".join(sorted(VALID_FILLER_FORMATS))
        raise ValueError(f"filler_format must be one of: {valid_formats}")
    
    all_results = []
    # Go through each subfolder in the corpus folder
    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)

        files = [
            f for f in os.listdir(subfolder_path)
            if f.endswith((".conllu", ".txt"))
        ]
        args = [
            (subfolder_path, f, pattern, target_lemma, target_pos, deprel, filler_format)
            for f in files
        ]

        with Pool(num_procs) as pool:
            for file_res in pool.imap_unordered(process_file, args, chunksize=10):
                all_results.extend(file_res)

    return pd.DataFrame(all_results, columns=["file", "sentence", "sfillers", "path"])
