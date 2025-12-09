import re
import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional
from SynFlow.utils import build_graph
from .const import DEFAULT_PATTERN

def find_by_path(graph, id2wordpos, id2deprel, tgt_ids, rel):
    """
    Finds all paths in a dependency graph starting from a set of target IDs that match
    a specified sequence of dependency relations.

    Args:
        graph (dict): Adjacency list representation of the dependency graph.
                      Keys are parent IDs, values are lists of child IDs.
        id2wordpos (dict): Maps token ID to its "lemma/POS" string.
        id2deprel (dict): Maps (parent_id, child_id) tuple to dependency relation string.
        tgt_ids (list): A list of target token IDs.
        rel (str): A single relation path string, e.g., "chi_obl > chi_case".
                   '>' separates sequential steps.

    Returns:
        list: A list of tuples, where each tuple contains:
              - List of "lemma/POS" strings for the context nodes found along the path.
              - The actual path string found (e.g., "chi_obl > chi_case").
              Returns an empty list if no path is found.
    """
    seq = [r.strip() for r in rel.split('>')]
    N   = len(seq)
    out = []

    def dfs(node, depth, seen, path_rels, path_nodes):
        if depth == N:
            out.append(( [id2wordpos[n] for n in path_nodes],
                         " > ".join(path_rels) ))
            return
        want = seq[depth]
        for nb in graph[node]:
            if nb in seen:                # ← block revisits, including target
                continue
            lbl = id2deprel.get((node, nb))
            if lbl == want:
                dfs(nb,
                    depth+1,
                    seen | {nb},
                    path_rels + [lbl],
                    path_nodes + [nb])

    for t in tgt_ids:
        dfs(t, 0, {t}, [], [])

    return out

def process_file(
        args: Tuple[str, str, Optional[re.Pattern], str, str, str]
        ) -> List[Tuple[str, str, List[str], str]]:
    """
    args = (corpus_folder, fname, pattern, target_lemma, target_pos, rel)
    returns list of (filename, sentence, ctx_nodes, path_str)
    """
    corpus_folder, fname, pattern, target_lemma, target_pos, rel = args
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
                    sentence_text = " ".join(sent_forms)
                    target_lp = f"{target_lemma}/{target_pos}"
                    tgt_ids = [tid for tid, lp in id2wp.items() if lp == target_lp]
                    for ctx_nodes, path_str in find_by_path(graph, id2wp, id2d, tgt_ids, rel):
                        # **Gắn thêm fname** vào đầu tuple
                        results.append((fname, sentence_text, ctx_nodes, path_str))

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
                 rel: str          = None,
                 num_processes: int= max(1, cpu_count() - 1)
                ) -> List[Tuple[str, str, List[str], str]]:
    """
    Walks corpus_folder in parallel, returns all (filename, sentence, ctx_nodes, path_str).
    """
    pattern       = pattern or DEFAULT_PATTERN
    num_procs     = num_processes or max(1, cpu_count()-1)
    
    all_results = []
    # Go through each subfolder in the corpus folder
    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)

        files = [
            f for f in os.listdir(subfolder_path)
            if f.endswith((".conllu", ".txt"))
        ]
        args = [
            (subfolder_path, f, pattern, target_lemma, target_pos, rel)
            for f in files
        ]

        with Pool(num_procs) as pool:
            for file_res in pool.imap_unordered(process_file, args, chunksize=10):
                all_results.extend(file_res)

    return all_results

