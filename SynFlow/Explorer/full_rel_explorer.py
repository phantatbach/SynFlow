import re
import os
from multiprocessing import Pool, cpu_count
import pandas as pd
from typing import List, Tuple, Dict, Set, Optional
from SynFlow.utils import build_graph, format_filler
from SynFlow.const import DEFAULT_PATTERN, VALID_FILLER_FORMATS, SENT_ID_PATTERN

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

# Find a single, sequential path
# It will be called multiple times by process_file.
def find_by_path(graph: Dict[int, List[int]], id2context: Dict[int, tuple[str, str, str]],
                 id2deprel: Dict[Tuple[int, int], str], tgt_id: int,
                 single_path_pattern: str, filler_format: str) -> List[Tuple[List[str], str]]:
    """
    Finds paths in the dependency graph starting from a single target ID that match
    a specified sequence of relationships.

    Args:
        graph (dict): Adjacency list representation of the dependency graph.
                      Keys are parent IDs, values are lists of child IDs.
        id2context (dict): Maps token ID to its token, lemma, and POS fields.
        id2deprel (dict): Maps (parent_id, child_id) tuple to dependency relation string.
        tgt_id (int): A single ID of the target token.
        single_path_pattern (str): A single relation path string, e.g., "chi_obl > chi_case".
                                   '>' separates sequential steps.

    Returns:
        list: A list of tuples, where each tuple contains:
              - List of formatted context fillers found along the path.
              - The actual path string found (e.g., "chi_obl > chi_case").
              Returns an empty list if no path is found.
    """
    # Split the single_path_pattern into sequential steps.
    seq_steps = [r.strip() for r in single_path_pattern.split(">")]
    num_steps = len(seq_steps)
    results: List[Tuple[List[str], str]] = []

    def dfs(node: int, depth: int, seen: Set[int], path_rels: List[str], path_nodes: List[int]):
        """
        Depth-first search to find a single path matching the relation sequence.
        """
        # Base case: If we have successfully traversed all required steps.
        if depth == num_steps:
            # Join the actual relations found into a path string.
            actual_path_str = " > ".join(path_rels)
            results.append((path_nodes, actual_path_str))
            return

        # Get the allowed relations for the current step.
        current_step_rels_str = seq_steps[depth]

        # Explore neighbors of the current node.
        # Ensure 'graph[node]' is handled safely if 'node' is not in graph (e.g., leaf node)
        for nb in graph.get(node, []): # Use .get() to avoid KeyError if node has no children
            # Skip if the neighbor has already been visited in this path to prevent cycles.
            if nb in seen:
                continue

            # Get the dependency label between the current node and its neighbor.
            lbl = id2deprel.get((node, nb))

            # Check if the actual label 'lbl' is one of the allowed relations for the current step.
            if lbl in current_step_rels_str:
                token, lemma, pos = id2context[nb]
                filler = format_filler(token, lemma, pos, lbl, filler_format)
                # Recursively call DFS for the neighbor, incrementing depth and updating path.
                dfs(nb,
                    depth + 1,
                    seen | {nb},             # Add neighbor to seen set for the next call
                    path_rels + [lbl],       # Add the actual label found
                    path_nodes + [filler])   # Add the formatted filler to the path nodes

    # Start DFS from the single target ID.
    # Initial call:
    # - 'tgt_id': starting node.
    # - 0: starting at depth 0 of the relation sequence.
    # - {tgt_id}: initially, only the target node is seen for this path.
    # - []: empty list for path relations (will be filled as relations are found).
    # - []: empty list for path nodes (will be filled with nodes reached after the target).
    dfs(tgt_id, 0, {tgt_id}, [], [])

    return results

def _find_all_unique_paths(graph: Dict[int, List[int]], id2deprel: Dict[Tuple[int, int], str],
                           tgt_id: int, max_path_depth: int) -> Set[str]:
    """
    Finds all unique paths in a dependency graph starting from a single target ID
    up to a given maximum depth.

    Args:
        graph (dict): Adjacency list representation of the dependency graph.
                      Keys are parent IDs, values are lists of child IDs.
        id2deprel (dict): Maps (parent_id, child_id) tuple to dependency relation string.
        tgt_id (int): A single ID of the target token.
        max_path_depth (int): Maximum depth of paths to search for.

    Returns:
        set: A set of strings, each representing a unique path found in the graph.
             Each string is a sequence of dependency relations joined by ' > '.
    """
    out: Set[str] = set()
    def dfs(node: int, depth: int, seen: Set[int], rel_path: List[str]):
        if depth == max_path_depth:
            out.add(" > ".join(rel_path))
            return

        has_child = False
        for nb in graph.get(node, []):
            if nb in seen:
                continue
            lbl = id2deprel.get((node, nb))
            if not lbl:
                continue
            has_child = True
            dfs(nb, depth + 1, seen | {nb}, rel_path + [lbl])

        if not has_child and rel_path:
            out.add(" > ".join(rel_path))

    dfs(tgt_id, 0, {tgt_id}, [])
    return out

def process_file(
        args: Tuple[str, str, Optional[re.Pattern], str, str, str, str, str]
        ) -> List[Tuple[str, str, List[Tuple[List[str], str]]]]:
    """
    Processes a single file to find sentences matching the given criteria.
    Now supports multiple independent path patterns and 'open'/'close'/'closeh' search_mode.

    Args:
        args (tuple): A tuple containing:
            - corpus_folder (str)
            - fname (str)
            - pattern (re.Pattern)
            - target_lemma (str)
            - target_pos (str)
            - deprel (str): The combined relation path string.
            - search_mode (str): 'open' (default) or 'close' or 'closeh'.
                'open': Match targets that include at least all required paths; extra slots and deeper specialisations allowed.
                'close': Match targets whose paths equal the required set exactly; no extra slots and no deeper specialisations.
                'closeh': Match targets with exactly the required slots; deeper specialisations under those slots allowed.
            - filler_format (str): Format for context fillers.

    Returns:
        List[Tuple[str, str, List[Tuple[List[str], str]]]]: A list of results.
    """
    corpus_folder, fname, pattern, target_lemma, target_pos, rel_combined, search_mode, filler_format = args
    pattern = pattern or DEFAULT_PATTERN
    results: List[Tuple[str, str, List[Tuple[List[str], str]]]] = []

    # Parse the combined 'deprel' string into individual required path patterns
    required_path_patterns_list = [p.strip() for p in rel_combined.split(" & ") if p.strip()]
    required_path_patterns_set = set(required_path_patterns_list) # For faster lookup and comparison

    if not required_path_patterns_list:
        return results
    
    has_target = False
    has_target_check_string = f"\t{target_lemma}\t{target_pos}"

    filepath = os.path.join(corpus_folder, fname)
    try:
        with open(filepath, encoding="utf8") as fh:
            sent_tokens, sent_forms = [], [] # Init for the whole file. Sent_tokens = lines, sent_forms = word forms only

            for line in fh:
                line = line.rstrip("\n")

                # Start a new sentence
                if line.startswith("<s id"):
                    has_target = False # Reset for new sentence
                    sent_tokens, sent_forms = [], [] # Reset for new sentence

                    # Get sentence ID
                    match = SENT_ID_PATTERN.match(line)
                    sent_id = match.group(1) if match else None
                
                # End of current sentence, build graph
                elif line.startswith("</s>"):
                    # Check if the sentence contains the target lemma/POS
                    if sent_tokens and has_target == True:
                        id2wp, graph, id2d = build_graph(sent_tokens, pattern)
                        id2context = build_context_lookup(sent_tokens, pattern)
                        sentence_text = " ".join(sent_forms)
                        target_lp = f"{target_lemma}/{target_pos}"

                        # Handle multiple target in 1 sentences
                        tgt_ids = [tid for tid, lp in id2wp.items() if lp == target_lp]

                        for current_target_id in tgt_ids:
                            all_paths_found_for_this_target: bool = True # Check if all required paths are present
                            found_paths_details: List[Tuple[List[str], str]] = [] # Stores results for each required path

                            for single_path_pattern in required_path_patterns_list: # Loop through each path in the list
                                current_path_results = find_by_path(graph, id2context, id2d, current_target_id, single_path_pattern, filler_format) # Find the path that matches the path pattern
                                if not current_path_results: # If the path is not found, stop
                                    all_paths_found_for_this_target = False
                                    break
                                found_paths_details.extend(current_path_results) # If the path is found, add it to the results

                            if all_paths_found_for_this_target: # If at least all required paths are present
                                if search_mode == "close":
                                    # For 'close' search_mode, we need to check if ONLY the required PATHS are present.
                                    # No horizontal expansion and no vertical specialisation are allowed

                                    # Derive max depth from the rel‐patterns themselves
                                    depths = [ len(p.split(">")) for p in required_path_patterns_list ]
                                    max_check_depth = max(depths)
                                    all_unique_paths_from_target = _find_all_unique_paths(
                                        graph, id2d, current_target_id, max_check_depth # Use the new argument here
                                    )

                                    # Check if the set of all found paths (up to max_check_depth)
                                    # is exactly equal to the set of required paths.
                                    if all_unique_paths_from_target == required_path_patterns_set:
                                        results.append((sent_id, sentence_text, found_paths_details))

                                elif search_mode == "closeh":
                                    # For 'closeh' search_mode, we need to check if ONLY the required SLOTS are present.
                                    # No horizontal expansion is allowed but the slots can be vertically specialised.

                                    # 1) compute required first‐hop labels
                                    required_horizontals = {
                                        p.split(">")[0].strip()
                                        for p in required_path_patterns_list
                                    }
                                    # 2) collect the actual first‐hop labels from this target
                                    actual_direct = {
                                        id2d.get((current_target_id, nb))
                                        for nb in graph.get(current_target_id, [])
                                        if id2d.get((current_target_id, nb))
                                    }
                                    # 3) only accept if they match exactly
                                    if actual_direct == required_horizontals:
                                        results.append((sent_id, sentence_text, found_paths_details))
                                    
                                elif search_mode == "open":
                                    # 'open' search_mode: if all required paths are found, it's a match.
                                    results.append((sent_id, sentence_text, found_paths_details))

                    sent_tokens, sent_forms = [], []

                else:
                    sent_tokens.append(line)
                    m = pattern.match(line)
                    if m:
                        sent_forms.append(m.group(1))
                    
                    # Check for target lemma/POS in the current line
                    if has_target_check_string in line:
                        has_target = True

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
    return results


def full_rel_explorer(corpus_folder: str,
                 pattern: re.Pattern = None,
                 target_lemma: str = None,
                 target_pos: str   = None,
                 deprel: str          = None,
                 search_mode: str         = "open",
                 filler_format: str = "lemma/pos",
                 num_processes: int= max(1, cpu_count() - 1)
                ) -> List[Tuple[str, str, List[Tuple[List[str], str]]]]:
    """
    Walks corpus_folder in parallel to find sentences matching the given criteria.
    Now supports finding sentences that satisfy ALL independent path patterns
    defined in 'deprel', with 'open' or 'close' matching search_mode.

    Args:
        corpus_folder (str): The path to the corpus directory.
        pattern (re.Pattern, optional): The regex pattern for parsing lines.
                                       Defaults to DEFAULT_PATTERN.
        target_lemma (str, optional): The lemma of the target word to search for.
        target_pos (str, optional): The POS tag of the target word.
        deprel (str, optional): The combined relation path string, e.g., "chi_obl > chi_case & chi_nsubj & chi_nobj".
                             ' & ' separates independent required paths. '>' separates sequential steps within a path.
        search_mode (str): 'open' (default) or 'close' or 'closeh'.
            'open': Match targets that include at least all required paths; extra slots and deeper specialisations allowed.
            'close': Match targets whose paths equal the required set exactly; no extra slots and no deeper specialisations.
            'closeh': Match targets with exactly the required slots; deeper specialisations under those slots allowed.
        filler_format: Format for context fillers. Must be one of
            ``"lemma_only"``, ``"lemma/pos"``, ``"lemma/deprel"``,
            ``"token_only"``, ``"token/pos"``, or ``"token/deprel"``.
        num_processes (int, optional): Number of processes to use for parallel processing.
                                       Defaults to (CPU count - 1) or 1 if CPU count is 1.

    Returns:
        List[Tuple[str, str, List[Tuple[List[str], str]]]]: A list of all found results across all files.
                                                Each result is:
                                                (filename, sentence_text, [ (ctx_nodes_path1, path_str1), ... ])
    """
    if not os.path.isdir(corpus_folder):
        print(f"Error: Corpus folder '{corpus_folder}' does not exist.")
        return []
    if target_lemma is None or target_pos is None or deprel is None:
        print("Error: 'target_lemma', 'target_pos', and 'deprel' must be provided.")
        return []
    if search_mode not in ["open", "close", "closeh"]:
        print("Error: 'search_mode' must be 'open' or 'close' or 'closeh'.")
        return []
    if filler_format not in VALID_FILLER_FORMATS:
        valid_formats = ", ".join(sorted(VALID_FILLER_FORMATS))
        print(f"Error: 'filler_format' must be one of: {valid_formats}.")
        return []

    pattern       = pattern or DEFAULT_PATTERN
    num_procs     = max(1, num_processes)
    all_results: List[Tuple[str, str, List[Tuple[List[str], str]]]] = []
    
    # Go through each subfolder in the corpus folder
    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)
        # Go through each file in each subfolder
        files = [
            f for f in os.listdir(subfolder_path)
            if f.endswith((".conllu", ".txt"))
        ]
        if not files:
            print(f"No .conllu or .txt files found in '{subfolder_path}'.")
            return []
        args = [
            (subfolder_path, f, pattern, target_lemma, target_pos, deprel, search_mode, filler_format)
            for f in files
        ]
        # Process the files in parallel
        with Pool(num_procs) as pool:
            for file_res in pool.imap_unordered(process_file, args, chunksize=10):
                all_results.extend(file_res)

    rows = []

    for sent_id, sentence, matches in all_results:
        for sfillers, path in matches:
            rows.append({
                "sentence_id": sent_id,
                "sentence": sentence,
                "sfillers": sfillers,
                "path": path
            })

    return pd.DataFrame(rows, columns=["sentence_id", "sentence", "sfillers", "path"])
