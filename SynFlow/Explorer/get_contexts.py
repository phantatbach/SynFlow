import os
import re
import pandas as pd
from typing import List, Dict, Optional
from .const import DEFAULT_PATTERN

def corpus_handler(file: str, pattern: Optional[re.Pattern] = None) -> List[List[Dict]]:
    """
    Read one CoNLL‐style file and split into sentences.
    Each sentence is a list of token‐dicts, where each dict has:
      {
        "line_num": int,    # 1-based line in the file
        "form": str,
        "lemma": str,
        "pos": str,
        "id": str,          # UD token ID
        "head": str,
        "deprel": str
      }
    """
    pattern = pattern or DEFAULT_PATTERN

    sentences = []
    current = []
    with open(file, encoding="utf8") as fh:
        line_no = 0
        for raw in fh:
            line_no += 1
            line = raw.rstrip("\n")
            if line.startswith("<s id"):
                # start a new sentence
                current = []
            elif line.startswith("</s>"):
                # end of current sentence → store it
                if current:
                    sentences.append(current)
                current = []
            else:
                # attempt to match a token line
                m = pattern.match(line)
                if not m:
                    continue
                form, lemma, pos, tid, head, deprel = m.groups()
                current.append({
                    "line_num": line_no,
                    "form": form,
                    "lemma": lemma,
                    "pos": pos,
                    "id": tid,
                    "head": head,
                    "deprel": deprel
                })
    return sentences

def get_contexts(
    slots_df: pd.DataFrame,
    corpus_path: str,
    output_path: str,
    pattern: Optional[re.Pattern] = None,
) -> pd.DataFrame:
    """
    Given slots_df (indexed by strings "target/file/line"), look up each file only once,
    extract the sentence that contains the target token, and save a new DataFrame
    with an added column "context" (the full sentence as a space‐joined string of FORM).

    Args:
      slots_df    : DataFrame whose index is "lemma/filename/line_num".
                    The columns are slot‐names ("chi_nsubj", etc.).
      corpus_path : path to the folder containing those filenames.
      pattern     : same regex used in corpus_handler for token lines.
      output_path : where to write the new CSV.

    Returns the new DataFrame.
    """
    pattern = pattern or DEFAULT_PATTERN
    
    # We'll build a new DataFrame that starts with slots_df and adds "context"
    df = pd.DataFrame(index=slots_df.index) # Take only the index of slots_df
    # df = slots_df.copy() # Keep all the slots
    df["context"] = ""   # initialize empty
    
    # We only want to load each file once, so we keep a simple cache:
    cache: Dict[str, List[List[Dict]]] = {}
    
    for idx in df.index:
        # idx is something like "run/fic_1922_4408_cleaned.txt_part15.txt/40"
        try:
            target_lemma, fname, line_str = idx.split("/", 2)
        except ValueError:
            raise ValueError(f"Index '{idx}' is not in 'lemma/filename/line' format.")
        line_num = int(line_str)
        
        file_path = os.path.join(corpus_path, fname)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found for index '{idx}'.")
        
        # load sentences from this file (once)
        if fname not in cache:
            cache[fname] = corpus_handler(file_path, pattern)
        sentences = cache[fname]
        
        # find the sentence that contains a token with line_num
        sentence_text = ""
        for sent in sentences:
            # Check if this sentence contains the token at the target line_num
            target_token_in_sent = None
            for tok in sent:
                if tok["line_num"] == line_num:
                    target_token_in_sent = tok
                    break
            
            if target_token_in_sent:
                # Build the sentence string, wrapping the target word
                forms = []
                for tok in sent:
                    if tok["line_num"] == line_num: # and tok["lemma"] == target_lemma
                        # We use line_num to uniquely identify the token, as lemma might appear multiple times
                        forms.append(f"<TAR> {tok['form']} <TAR>")
                    else:
                        forms.append(tok["form"])
                sentence_text = " ".join(forms)
                break # Found the sentence, no need to check others
        
        df.at[idx, "context"] = sentence_text
    
    # Finally, save output
    df.to_csv(output_path)
    print(f"Wrote contexts to {output_path}.")
    return df
