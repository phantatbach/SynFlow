import re
import csv
import os
from collections import Counter
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from SynFlow.utils import build_graph

DEFAULT_PATTERN = re.compile(
    r'([^\t]+)\t'      # FORM
    r'([^\t]+)\t'      # LEMMA
    r'([^\t]+)\t'      # POS
    r'([^\t]+)\t'      # ID
    r'([^\t]+)\t'      # HEAD
    r'([^\t]+)'        # DEPREL
)

# def find_paths(id2lp, graph, id2d, target_lp, max_length):
#     """
#     Finds all unique paths of dependency labels starting from tokens with a specific lemma/POS
#     in a dependency graph, with a maximum specified path length.

#     Args:
#         id2lp (dict): Mapping of token id to 'lemma/pos'.
#         graph (dict): Dependency graph mapping each token id to its neighbors.
#         id2d (dict): Mapping of edge (tuple of token ids) to dependency relation label.
#         target_lp (str): The target lemma/POS string to search paths from.
#         max_length (int): Maximum length of paths to explore.

#     Returns:
#         list: A list of unique path strings of dependency labels, where each path
#               is represented as labels joined by ' > '.
#     """

#     out = []
#     for t, lp in id2lp.items():
#         if lp != target_lp: continue

#         def dfs(node, depth, seen, rel_path):
#             '''
#             Recursively find paths of length <= max_length from a specific node with DFS.
#             '''
#             if 0 < depth <= max_length:
#                 out.append(" > ".join(rel_path)) # Append all items in rel_path with >
#                 # out.append((depth, " > ".join(rel_path)))
#             if depth == max_length:
#                 return
            
#             for nb in graph[node]: # Explore all the neighbours
#                 if nb in seen: # skip any node we've already visited (including t)
#                     continue
#                 lbl = id2d.get((node, nb))
#                 if not lbl:
#                     continue
#                 dfs(nb,
#                     depth+1,
#                     seen | {nb}, # Union of seen and neighbours
#                     rel_path + [lbl] # Add edge label to the path
#                 )

#         dfs(t, 0, {t}, [])

#     return out

def find_paths(id2lp, graph, id2d, target_lp, max_length):
    out = []

    for t, lp in id2lp.items():
        if lp != target_lp:
            continue

        def dfs(node, depth, seen, rel_path):
            if depth == max_length:
                out.append(" > ".join(rel_path))
                return

            has_child = False
            for nb in graph.get(node, []):
                if nb in seen:
                    continue
                lbl = id2d.get((node, nb))
                if not lbl:
                    continue

                has_child = True
                dfs(
                    nb,
                    depth + 1,
                    seen | {nb},
                    rel_path + [lbl]
                )

            # Nếu không còn child nào để đi, và path chưa đủ max_depth nhưng đây là leaf → cũng append
            if not has_child and rel_path:
                out.append(" > ".join(rel_path))

        dfs(t, 0, {t}, [])

    return out

def process_file(args) -> Counter:
    """
    Process a single file.

    Given a filename, a corpus folder, a regex pattern, a target lemma, a target POS,
    and a maximum path length, 
    read the file, build a dependency graph for each sentence,
    find all context paths (up to max_length) that start from any of the target ids,
    and count each distinct path.

    Returns a Counter object with the path counts.
    """
    corpus_folder, fname, pattern, target_lemma, target_pos, max_length = args
    ctr = Counter()
    path = os.path.join(corpus_folder, fname)

    with open(path, encoding="utf8") as fh:
        sent = []
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("<s id"):
                sent = []
            elif line.startswith("</s>"):

                # build graph when the whole sentence is appended
                id2lp, graph, id2d = build_graph(sent, pattern)
                target_lp = f"{target_lemma}/{target_pos}"

                # for each target token in this sentence
                for tid, lp in id2lp.items():
                    if lp != target_lp:
                        continue

                    # find all paths of length ≤ max_length
                    paths = find_paths(id2lp, graph, id2d, target_lp, max_length)
                    # get unique paths
                    unique = sorted(set(paths))

                    # build the full-pattern string:
                    #   target_lemma & > path1 & > path2 > path3 & ...
                    parts = [target_lemma] + ["> " + p for p in unique]
                    pattern_str = " & ".join(parts)

                    ctr[pattern_str] += 1
            else:
                sent.append(line)
    return ctr

def save_to_csv(counter, output_path="output.csv"):
    split_patterns = []
    max_slots = 0

    # Tách pattern và tính max slots
    for pattern_str, freq in counter.items():
        parts = pattern_str.split(" & ")
        target = parts[0]
        slots = parts[1:]
        split_patterns.append((freq, target, slots))
        if len(slots) > max_slots:
            max_slots = len(slots)

    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='&')

        # Header
        header = ["Frequency", "Target"] + [f"Slot{i+1}" for i in range(max_slots)]
        writer.writerow(header)

        # Ghi từng row, pad nếu slot thiếu
        for freq, target, slots in sorted(split_patterns, key=lambda x: -x[0]):
            row = [freq, target] + slots + [""] * (max_slots - len(slots))
            writer.writerow(row)

    print(f"CSV saved to {output_path}")

def arg_comb_explorer(
    corpus_folder: str,
    target_lemma: str,
    target_pos: str,
    output_folder: str,
    max_length: int = 1,
    top_n: int = 20,
    num_processes: int = None,
    pattern: re.Pattern = None
) -> Counter:
    pattern   = pattern or DEFAULT_PATTERN
    num_procs = num_processes or max(1, cpu_count()-1)

    files = [f for f in os.listdir(corpus_folder)
             if f.endswith((".conllu", ".txt"))]
    args = [
        (corpus_folder, f, pattern,
         target_lemma, target_pos, max_length)
        for f in files
    ]

    total = Counter()
    with Pool(num_procs) as pool:
        for file_ctr in pool.imap_unordered(process_file, args, chunksize=10):
            total.update(file_ctr)

    print(f"Total instances: {sum(total.values())}, distinct patterns: {len(total)}")
    if total:
        labels, freqs = zip(*total.most_common(top_n))
        plt.figure(figsize=(min(12, 0.3*len(labels)), 6))
        plt.bar(range(len(freqs)), freqs)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.ylabel("Count")
        plt.title(f"Top {top_n} unique combinations around {target_lemma}/{target_pos} (≤{max_length}-hop)")
        plt.tight_layout()
        plt.show()

        save_to_csv(total, output_path=f"{output_folder}/{target_lemma}_arg_comb_{max_length}_hops.csv")
    return total

