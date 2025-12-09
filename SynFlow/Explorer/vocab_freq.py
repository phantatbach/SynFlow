import os
from collections import Counter
from multiprocessing import Pool, cpu_count

def count_lemma_file(path, mode):
    """
    Count frequencies according to `mode`:
      - 'lemma/pos_init': lemma + "\t" + POS‐initial
      - 'lemma/deprel':     lemma + "\t" + deprel
      - 'lemma/pos':        lemma + "\t" + full POS
    """
    local_lemma_counts = Counter()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('<s') or line.startswith('</s>'):
                continue
            
            # Split the lines into different parts
            parts = line.split()
            if len(parts) < 6:
                continue
            # Default: wordform, lemma, pos, id, head, deprel
            lemma, pos, deprel = parts[1], parts[2], parts[5]
            pos_initial = pos[0]

            if mode == 'lemma_pos_init':
                key = f"{lemma}/{pos_initial}"
            elif mode == 'lemma_deprel':
                key = f"{lemma}/{deprel}"
            elif mode == 'lemma_pos':
                key = f"{lemma}/{pos}"
            else:
                raise ValueError(f'Mode must be one of "lemma_pos_init", "lemma_deprel", or "lemma_pos", got {mode}')

            # Count
            local_lemma_counts[key] += 1

    return local_lemma_counts

def count_lemma_parallel_subfolder(subfolder_path, file_ext=None, mode='lemma_pos'):
    # Get list of file
    all_files = []
    for root, _, files in os.walk(subfolder_path):
        for fname in files:
            if file_ext and not fname.endswith(file_ext):
                continue
            all_files.append(os.path.join(root, fname))
    
    # Parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(count_lemma_file, [(path, mode) for path in all_files])

    # Combine counters
    total_counter = Counter()
    for counter in results:
        total_counter.update(counter)

    return dict(total_counter)


def save_freqs(freq_dict, out_folder, subfolder, mode='lemma_pos'):
    """
    Write out `<mode>_freq.txt` into out_folder.
    Each line: key<TAB>frequency
    """
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{subfolder}_{mode}_freq.txt")
    with open(out_path, 'w', encoding='utf-8') as out:
        for key, freq in sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True):
            out.write(f"{key}\t{freq}\n")
    return out_path


def gen_lemma_freq(corpus_path, out_folder, file_ext=None, mode='lemma_pos'):
    """
    Complete pipeline: count then save.
    Returns the path to the file written.
    """
    for subfolder in os.listdir(corpus_path):
        subfolder_path = os.path.join(corpus_path, subfolder)
        freqs = count_lemma_parallel_subfolder(subfolder_path, file_ext, mode)
        save_freqs(freqs, out_folder, subfolder, mode)

def analyze_single_subcorpus_vocab(filepath):
    """
    Analyzes a single corpus file.
    
    Returns:
        (int): Token count for this file.
        (set): Unique types (e.g., 'to/A') for this file.
        (set): Unique words (e.g., 'to') for this file.
    """
    file_tokens = 0
    file_unique_types = set()
    file_unique_words = set()

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                
                if len(parts) >= 2:
                    word_tag_pair = parts[0]
                    count_str = parts[-1]
                    
                    try:
                        file_tokens += int(count_str)
                        file_unique_types.add(word_tag_pair)
                        
                        word = word_tag_pair.split('/')[0]
                        file_unique_words.add(word)
                        
                    except ValueError:
                        print(f"Skipping malformed line in {filepath}: {line}", file=sys.stderr)
                
                else:
                    print(f"Skipping short line in {filepath}: {line}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: File not found {filepath}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)

    return file_tokens, file_unique_types, file_unique_words

def analyze_corpus_vocab(corpus_vocab_directory):
    """
    Walks through a directory and analyzes all .txt files found within it
    and its subdirectories.
    
    Args:
        root_directory (str): The path to the main corpus folder.
    """
    results = dict()

    # Corpus-wide aggregates
    grand_total_tokens = 0
    aggregate_unique_types = set()
    aggregate_unique_words = set()
    
    file_count = 0

    print(f"Starting analysis of: {corpus_vocab_directory}\n")

    # os.walk travels through all subdirectories
    for dirpath, _, filenames in os.walk(corpus_vocab_directory):
        for filename in filenames:
            # Check if the file is a .txt file
            if filename.endswith(".txt"):
                file_count += 1
                filepath = os.path.join(dirpath, filename)
                
                # Analyze the single file
                tokens, types, words = analyze_single_subcorpus_vocab(filepath)
                results[filename] = {'tokens': tokens, 'types': len(types), 'words': len(words)}

                # Add this file's stats to the grand totals
                grand_total_tokens += tokens
                
                # .update() adds all items from one set to another
                aggregate_unique_types.update(types)
                aggregate_unique_words.update(words)

    if file_count == 0:
        print("No .txt files were found in that directory.")
        return

    results['All'] = {
        'total_tokens': grand_total_tokens,
        'types': len(aggregate_unique_types),
        'words': len(aggregate_unique_words)
    }

    return results