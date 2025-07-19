import re
import os
import json
from collections import Counter, deque
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from SynFlow.utils import build_graph

# ——— Default token‐pattern (you can override this) ————————————————
DEFAULT_PATTERN = re.compile(
    r'([^\t]+)\t'      # word form
    r'([^\t]+)\t'      # lemma
    r'([^\t]+)\t'      # POS (UPOS or XPOS)
    r'([^\t]+)\t'      # ID
    r'([^\t]+)\t'      # HEAD
    r'([^\t]+)'        # DEPREL
)

def get_contexts(graph, id2lemma_pos, id2deprel, tgt_ids, max_length):
    """
    Given a dependency graph, a mapping of id to lemma/pos, a mapping of edge to deprel,
    a list of target ids, and a maximum length, find all context argument paths (up to max_length)
    that start from any of the target ids. Use a breadth‐first search.

    Returns a list of context paths, where each path is a string of dependency labels
    joined by ' > '.
    """
    contexts = [] # Create an empty list to save the context paths
    for t in tgt_ids:
        '''
        Breadth-first search
        Create a double-ended queue for a list of neightbours and a set to keep track of visited nodes
        Items in the queue have the following format: (node, depth, arg_path to the node)
        For each node in the queue, get the neighbours and depths and add them to the queue.
        '''
        q, visited = deque([(t, 0, [])]), {t}
        while q:
            node, depth, arg_path = q.popleft() # Continue as long as there are nodes to visit
            if depth == max_length:
                continue # Skip to the next item in the queue if we've reached the maximum path length
            for nb in graph[node]: # For each neighbour
                if nb in visited: # Prevent revisiting the same node
                    continue
                lbl = id2deprel[(node, nb)] # Get the edge label from node to neighbour
                new_path = arg_path + [lbl]
                contexts.append(' > '.join(new_path))

                visited.add(nb)
                q.append((nb, depth + 1, new_path))
    return contexts

def process_file(args):
    """
    Process a single file.

    Given a filename, a corpus folder, a regex pattern, a target lemma, a target POS,
    and a maximum path length, 
    read the file, build a dependency graph for each sentence,
    find all context argument paths (up to max_length) that start from any of the target ids,
    and count each distinct path.

    Returns a Counter object with the path counts.
    """
    fname, corpus_folder, pattern, target_lemma, target_pos, max_length = args
    counter = Counter()
    path = os.path.join(corpus_folder, fname)
    with open(path, encoding='utf8') as fh:
        sentence = []
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('<s id'):
                sentence = []
            elif line.startswith('</s>'):

                # Build a dependency graph when the whole sentence is appended
                id2lp, graph, id2dep = build_graph(sentence, pattern)
                # Find words with the target lemma and POS
                tgt_ids = [
                    idx for idx, lp in id2lp.items()
                    if lp.split('/')[0] == target_lemma
                       and lp.split('/')[1] == target_pos
                ]
                # If match then find contexts
                if tgt_ids:
                    for p in get_contexts(graph, id2lp, id2dep, tgt_ids, max_length):
                        counter[p] += 1

            else:
                sentence.append(line)
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
    plt.title(f'Top {top_n} arguments of “{target_lemma}” (max_length={max_length})')
    plt.tight_layout()
    plt.show()

def arg_explorer(
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
    Walks your folder in parallel, collects argument‐counts around target tokens,
    plots and returns the aggregated Counter.

    Args:
      corpus_folder     – path to your .conllu/.txt files
      target_lemma      – lemma to look for (e.g. 'run')
      target_pos        – POS of target (e.g. 'v' or 'n')
      max_length        – how many hops in the undirected graph
      top_n             – how many top arguments to plot
      num_processes     – None (auto) or int
      pattern           – custom regex for your token lines
    """
    pattern        = pattern or DEFAULT_PATTERN
    num_processes  = num_processes or max(1, cpu_count() - 1)

    # gather filenames
    files = [
        f for f in os.listdir(corpus_folder)
        if f.endswith(('.conllu', '.txt'))
    ]

    # prepare per‐file args tuples
    args_list = [
        (f, corpus_folder, pattern,
         target_lemma, target_pos, max_length)
        for f in files
    ]

    global_counter = Counter()
    with Pool(num_processes) as pool:
        for ctr in pool.imap_unordered(process_file, args_list, chunksize=10):
            global_counter.update(ctr)

    print(f'Collected {sum(global_counter.values())} context links, '
          f'{len(global_counter)} distinct arguments.')
    plot_dist(global_counter, target_lemma, max_length, top_n)

    # SAVE COUNTER AS DICT
    sorted_args = dict(sorted(global_counter.items(), key=lambda x: x[1], reverse=True))
    output_path = os.path.join(output_folder, f'{target_lemma}_{target_pos}_arguments.json')
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(sorted_args, f_out, ensure_ascii=False, indent=2)
    print(f'Saved path frequencies to: {output_path}')

    return global_counter
