import re
import csv
import os
from collections import Counter
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from SynFlow.utils import build_graph
from typing import Dict
from .const import DEFAULT_PATTERN

def find_paths_from(graph, id2d, start_id, max_length):
    out = []

    def dfs(node, depth, seen, rel_path):
        if depth == max_length:
            if rel_path:                     # tránh append chuỗi rỗng
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
            dfs(nb, depth + 1, seen | {nb}, rel_path + [lbl])

        if not has_child and rel_path:
            out.append(" > ".join(rel_path))

    dfs(start_id, 0, {start_id}, [])
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

    has_target = False
    has_target_check_string = f'\t{target_lemma}\t{target_pos}'
    
    with open(path, encoding="utf8") as fh:
        sent_tokens = []
        for line in fh:
            line = line.rstrip("\n")

            # Start a new sentence
            if line.startswith("<s id"):
                sent_tokens = []
                has_target = False  # Reset for new sentence

            elif line.startswith("</s>"):
                if has_target and sent_tokens:
                    # build graph when the whole sentence is appended
                    id2lp, graph, id2d = build_graph(sent_tokens, pattern)
                    target_lp = f"{target_lemma}/{target_pos}"

                    # for each target token in this sentence
                    for tid, lp in id2lp.items():
                        if lp != target_lp:
                            continue

                        paths  = find_paths_from(graph, id2d, tid, max_length)  # <— dùng tid to get all the path from each target token
                        unique = sorted(set(paths)) # Take only 1 type of slot for each token. Need to think about cases where there are duplicate slots of the same token (e.g., big bad wolf)

                        parts = [target_lemma] + ["> " + p for p in unique]
                        pattern_str = " & ".join(parts)
                        ctr[pattern_str] += 1
            else:
                sent_tokens.append(line)
                # Check for target lemma/POS in the current line
                if has_target_check_string in line:
                    has_target = True
    return ctr

def save_to_csv_with_subfolder(rows, output_path="output.csv"):
    """
    rows: iterable of (subfolder, freq, target, [slots...])
    Ghi một CSV duy nhất, delimiter '&', có cột Subfolder.
    """
    # Tính max số slot để pad
    max_slots = 0
    for _, _, _, slots in rows:
        if len(slots) > max_slots:
            max_slots = len(slots)

    # Ghi file
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='&')
        header = ["Subfolder", "Frequency", "Target"] + [f"Slot{i+1}" for i in range(max_slots)]
        writer.writerow(header)

        # Sắp xếp: subfolder rồi freq giảm dần
        rows_sorted = sorted(rows, key=lambda r: (r[0], -r[1], r[2]))
        for subf, freq, target, slots in rows_sorted:
            row = [subf, freq, target] + slots + [""] * (max_slots - len(slots))
            writer.writerow(row)

    print(f"CSV saved to {output_path}")


def spath_comb_explorer(
    corpus_folder: str,
    target_lemma: str,
    target_pos: str,
    output_folder: str,
    max_length: int = 1,
    top_n: int = 20,
    num_processes: int = None,
    pattern: re.Pattern = None
) -> Dict[str, Counter]:
    """
    Trả về dict{subfolder: Counter}, và vẽ top_n theo từng subfolder.
    Đồng thời ghi một CSV tổng hợp có cột Subfolder.
    """
    pattern   = pattern or DEFAULT_PATTERN
    num_procs = num_processes or max(1, cpu_count()-1)

    all_totals: Dict[str, Counter] = {}
    csv_rows = []  # sẽ chứa (subfolder, freq, target, slots)

    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        files = [f for f in os.listdir(subfolder_path)
                 if f.endswith((".conllu", ".txt"))]

        args = [
            (subfolder_path, f, pattern, target_lemma, target_pos, max_length)
            for f in files
        ]

        total = Counter()
        if args:
            with Pool(num_procs) as pool:
                for file_ctr in pool.imap_unordered(process_file, args, chunksize=10):
                    total.update(file_ctr)

        all_totals[subfolder] = total

        print(f"[{subfolder}] Total instances: {sum(total.values())}, distinct patterns: {len(total)}")

        # Vẽ top_n cho subfolder này
        if total:
            labels, freqs = zip(*total.most_common(top_n)) # Tuple unpacking then zipping to list for plotting
            plt.figure(figsize=(min(14, 0.35*len(labels)), 6))
            plt.bar(range(len(freqs)), freqs)
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.ylabel("Count")
            plt.title(f"{subfolder}: Top {top_n} unique combinations around {target_lemma}/{target_pos} (≤{max_length}-hop)")
            plt.tight_layout()
            plt.show()

            # Chuẩn bị row CSV cho mọi pattern của subfolder này
            for pattern_str, freq in total.items():
                parts  = pattern_str.split(" & ")
                target = parts[0]
                slots  = parts[1:] # This also contains '>'
                csv_rows.append((subfolder, freq, target, slots))

    # Ghi CSV tổng hợp
    os.makedirs(output_folder, exist_ok=True)
    out_csv = os.path.join(
        output_folder,
        f"{target_lemma}_{target_pos}_spath_combs_{max_length}_hops.csv"
    )
    save_to_csv_with_subfolder(csv_rows, output_path=out_csv)

    return all_totals

