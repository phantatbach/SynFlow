import re
import os
import json
from collections import Counter, deque
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from SynFlow.utils import build_graph
from .const import DEFAULT_PATTERN

def get_contexts(graph, id2d, tgt_ids, max_length):
    """
    Given a dependency graph, a mapping of id to lemma/pos, a mapping of edge to deprel,
    a list of target ids, and a maximum length, find all context slot paths (up to max_length)
    that start from any of the target ids. Use a breadth-first search.

    Returns a list of context paths, where each path is a string of dependency labels
    joined by ' > '.
    """
    out = [] # Create an empty list to save the context paths
    for t in tgt_ids:
        q = deque([(t, 0, [], {t})])  # seen riêng cho từng path
        while q:
            node, depth, path, seen = q.popleft() # seen riêng cho từng path
            if depth == max_length:
                continue # Skip to the next item in the queue if we've reached the maximum path length
            for nb in graph.get(node, []): # For each neighbour
                if nb in seen: # Prevent revisiting the same node in the same path
                    continue
                lbl = id2d.get((node, nb)) # Get the edge label from node to neighbour
                if not lbl:
                    continue
                new_path = path + [lbl]
                out.append(" > ".join(new_path))
                q.append((nb, depth + 1, new_path, seen | {nb}))
    return out

def process_file(args):
    """
    Process a single file.

    Given a filename, a corpus folder, a regex pattern, a target lemma, a target POS,
    and a maximum path length, 
    read the file, build a dependency graph for each sentence,
    find all context slot paths (up to max_length) that start from any of the target ids,
    and count each distinct path.

    Returns a Counter object with the path counts.
    """
    fname, corpus_folder, pattern, target_lemma, target_pos, max_length = args
    counter = Counter()
    path = os.path.join(corpus_folder, fname)

    has_target = False
    has_target_check_string = f'\t{target_lemma}\t{target_pos}'

    with open(path, encoding='utf8') as fh:
        sent_tokens = []
        for line in fh:
            line = line.rstrip('\n')

            # Start a new sentence
            if line.startswith('<s id'):
                sent_tokens = []
                has_target = False # Reset for new sentence
            
            # End of a sentence. Build graph and process if target found
            elif line.startswith('</s>'):
                if sent_tokens and has_target == True:
                    # Build a dependency graph when the whole sentence is appended
                    id2lp, graph, id2d = build_graph(sent_tokens, pattern)
                    # Find words with the target lemma and POS
                    tgt_ids = [
                        idx for idx, lp in id2lp.items()
                        if lp.split('/')[0] == target_lemma
                        and lp.split('/')[1] == target_pos
                    ]
                    # If match then find contexts
                    if tgt_ids:
                        for p in get_contexts(graph, id2d, tgt_ids, max_length):
                            counter[p] += 1

            else:
                sent_tokens.append(line)
                # Check for target lemma/POS in the current line
                if has_target_check_string in line:
                    has_target = True
    return counter

def plot_dist(counter, target_lemma, max_length, top_n):
    if not counter:
        print("Nothing to plot.")
        return
    labels, freqs = zip(*counter.most_common(top_n))
    plt.figure(figsize=(min(12, 0.3 * len(labels)), 6))
    plt.bar(range(len(freqs)), freqs)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} slot-paths of “{target_lemma}” (max_length={max_length})')
    plt.tight_layout()
    plt.show()

def spath_explorer(
    corpus_folder: str,
    target_lemma: str,
    target_pos: str,
    output_folder: str,
    max_length: int = 1,
    top_n: int = 20,
    num_processes: int = max(1, cpu_count() - 1),
    pattern: re.Pattern = None
):
    """
    Walks your folder in parallel, collects slot-path‐counts around target tokens,
    plots and returns the aggregated Counter.

    Args:
      corpus_folder     – path to your .conllu/.txt files
      target_lemma      – lemma to look for (e.g. 'run')
      target_pos        – POS of target (e.g. 'v' or 'n')
      max_length        – how many hops in the undirected graph
      top_n             – how many top slot-paths to plot
      num_processes     – None (auto) or int
      pattern           – custom regex for your token lines
    """
    pattern        = pattern or DEFAULT_PATTERN
    num_processes  = num_processes or max(1, cpu_count() - 1)
    all_results = {}  # <-- dict of dict

    # Go through each subfolder in the corpus folder
    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)

        # Gather filenames within each subfolder in the corpus folder
        files = [
            f for f in os.listdir(subfolder_path)
            if f.endswith(('.conllu', '.txt'))
        ]

        # prepare per‐file slot-paths tuples
        slotpaths_list = [
            (f, subfolder_path, pattern,
            target_lemma, target_pos, max_length)
            for f in files
        ]

        global_counter = Counter()
        with Pool(num_processes) as pool:
            for ctr in pool.imap_unordered(process_file, slotpaths_list, chunksize=10):
                global_counter.update(ctr)

        print(f'[{subfolder}] Collected {sum(global_counter.values())} context links, '
              f'{len(global_counter)} distinct arguments.')

        plot_dist(global_counter, target_lemma, max_length, top_n)

        # SAVE COUNTER AS DICT
        sorted_slotpaths = dict(sorted(global_counter.items(), key=lambda x: x[1], reverse=True))
        all_results[subfolder] = sorted_slotpaths  # <-- save by subfolder

    output_path = os.path.join(output_folder, f'{target_lemma}_{target_pos}_spaths.json')
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=2)
    print(f'Saved slot-path frequencies to: {output_path}')

    return global_counter
