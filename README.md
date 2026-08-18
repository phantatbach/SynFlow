# SynFlow

SynFlow is an end-to-end open-source Python toolkit for multidimensional diachronic
analysis of linguistic usage.

Lexical semantic change is often modelled with vector-space representations or
sense distributions. These approaches can be effective, but they usually require
substantial data and often give limited insight into which aspects of usage are
changing. Diachronic corpus linguistics, by contrast, often studies more
interpretable dimensions such as syntactic behaviour, morphological properties,
constructions, or semantic representations. These dimensions are usually handled
through separate workflows.

SynFlow provides a common workflow for these dimensions. It converts linguistic
observations into period-specific distributions, compares those distributions
with shared distance measures, and decomposes the observed changes into the
values that drive them.

The current public workflow is notebook-based. Users are expected to run the
provided notebooks, configure paths and target words inside those notebooks, and
use the existing SynFlow functions as they are. Core functions and internal
pattern definitions are implementation details and should not be edited for
normal use.

## Main Capabilities

SynFlow supports multidimensional diachronic analysis through:

- Stanza-based parsing of raw sentence files into SynFlow's parsed-corpus
  format;
- Tracking change in different linguistic dimensions, such as:
    - Individual dependency slots and their lexical fillers;
    - Constructional configurations;
    - Morphological types and morphological features;
- Value-level contribution analysis for interpreting what changed;
- Support weighting and permutation testing for more careful historical-corpus
  analysis.
- For lexical fillers, SynFlow further supports incremental clustering.

SynFlow was developed by [Bach Phan-Tat](https://phantatbach.github.io/).

## Installation

Clone the repository, create a virtual environment, and install the required
packages:

```bash
git clone <repo-url>
cd SynFlow
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The notebooks include the repository path in their setup cells, so no package
installation step is needed beyond installing `requirements.txt`.

Python 3.10 or newer is recommended.

## Requirements

The dependency list is maintained in `requirements.txt`. It includes:

- notebook runtime: `ipykernel`, `ipython`;
- data handling: `pandas`, `numpy`;
- statistics and distance measures: `scipy`, `statsmodels`, `scikit-learn`;
- plotting: `matplotlib`, `plotly`, `seaborn`, `dash`;
- corpus parsing: `stanza`, `tqdm`;
- embedding and clustering support: `gensim`, `umap-learn`.

Use this command whenever setting up a new environment:

```bash
python -m pip install -r requirements.txt
```

## Recommended Notebooks

The recommended entry points are in:

```text
notebooks/
```

Use these notebooks:

- `Stanza_Data_Prep.ipynb`
- `SynFlow_Slot.ipynb`
- `SynFlow_Constr.ipynb`
- `SynFlow_Morph.ipynb`
- `SynFlow_Qual_Insp.ipynb`
- `SynFlow_DiaEmb.ipynb`

Other notebooks and lower-level functions are not part of the recommended user
workflow at the moment.

### 1. Stanza Data Prep

Use `Stanza_Data_Prep.ipynb` to parse raw sentence files into SynFlow's
seven-field parsed-token format.

The raw input root should contain files inside subfolders:

```text
raw_root/
  1995-2000/
    file_001.txt
    file_002.txt
  2001-2005/
    file_001.txt
```

Each non-empty line is treated as one raw sentence. Files directly inside the
root folder are ignored. By default the parser reads `.txt`, `.conll`,
`.conllu`, and `.json` files. The notebook requires the user to choose the
Stanza language and model configuration explicitly, for example:

```python
language = "en"
model = "ewt"
processor_models = None
```

or:

```python
language = "de"
model = None
processor_models = {
    "tokenize": "gsd",
    "mwt": "gsd",
    "pos": "hdt",
    "lemma": "hdt",
    "depparse": "hdt",
}
```

### 2. Slot Level

Use `SynFlow_Slot.ipynb` to inspect syntactic slots and slot fillers distributions.

This notebook is for questions such as:

- Which slot types become more or less frequent over time?
- The change in the internal structures of different slots over time?
- Which periods show stronger changes in slot/slot-filler distribution?

### 3. Construction Level

Use `SynFlow_Constr.ipynb` to analyse constructional configurations.

This notebook is for questions such as:

- How does a target construction change across periods?
- Which constructional patterns become more or less frequent?
- How do construction-level distributions compare over time?

### 4. Morphology Level

Use `SynFlow_Morph.ipynb` to analyse morphological types and features.

This notebook is for questions such as:

- Do the morphological properties of a target change over time?
- Which feature types or feature values drive the observed change?
- Are changes concentrated in dimensions such as `Degree`, `Number`, `Case`, or
  `Tense`?

### 5. Qualitative Inspection

Use `SynFlow_Qual_Insp.ipynb` to inspect which values or components drive a
distributional change. This notebook can be used after any dimension-specific
analysis, including slot, construction, morphology, feature-value, or
slot-filler distributions.

This notebook is for questions such as:

- Which values increase, decrease, appear, or disappear over time?
- Which values contribute most to the distance between two periods?
- Which components are responsible for the observed change in a selected
  dimension?

This is the main notebook for qualitative interpretation of change drivers.

### 6. Diachronic Embeddings

Use `SynFlow_DiaEmb.ipynb` to prepare Word2Vec training data from parsed
corpora, train period-specific Word2Vec models, align those models, and inspect
slot fillers with diachronic embedding plots and clustering.

This notebook is for questions such as:

- How are fillers distributed in embedding space?
- Do fillers form interpretable clusters within a period?
- How do filler clusters develop incrementally across periods?
- Which fillers move between clusters?

Training data, models, and aligned models are written to user-configured local
paths. Large embedding artifacts are not included in this repository.

## Expected Data Layout

The Stanza data-preparation notebook expects raw sentence files inside
subfolders of an input root:

```text
raw_root/
  1995-2000/
    doc_001.txt
    doc_002.txt
  2001-2005/
    doc_001.txt
```

The parser mirrors this structure into the output root and writes parsed files
using sentence boundaries like:

```text
<s id=1995-2000_1>
The	the	DET	1	2	det	Definite=Def|PronType=Art
dog	dog	NOUN	2	3	nsubj	Number=Sing
barks	bark	VERB	3	0	root	Mood=Ind|Tense=Pres|VerbForm=Fin
</s>
```

Token lines are expected to contain seven tab-separated fields:

```text
TOKEN    LEMMA    POS    ID    HEAD    DEPREL    FEATS
```

Use parsed corpora that already follow this format. The notebooks and package
functions for slot, construction, morphology, qualitative inspection, and
embedding analysis assume this contract.

## End-to-End Workflow

SynFlow supports an end-to-end workflow from raw sentences to temporal and
embedding-based analysis:

```text
raw sentences
  -> Stanza parse
  -> SynFlow parsed corpus
  -> working with different dimensions:
       slot types/slot fillers
       constructions
       feature types/features
```

With slot-filler information, the workflow can continue into diachronic
embeddings:

```text
slot fillers
  -> W2V training data
  -> W2V models
  -> aligned embeddings
  -> SynFlow diachronic embedding analysis
```

The temporal-analysis workflow is shared across dimensions:

```text
target occurrences
  -> linguistic observations
  -> period-specific distributions
  -> distance scores
  -> value-level explanations
```

This makes it possible to compare different linguistic dimensions within a
common workflow instead of maintaining separate analyses for every dimension.

## Interpreting Change

SynFlow's temporal analysis uses distribution distances:

- `jsd`: Jensen-Shannon divergence;
- `cosine_distance`: one minus cosine similarity;
- `tvd`: total variation distance.

The same workflow can report:

- raw distance between periods;
- support-weighted distance for sparse data;
- item-level contributions to a distance score;
- permutation-test p-values;
- FDR-corrected q-values.

Contribution labels mark direction:

- `in_`: increased in the later period;
- `de_`: decreased in the later period;
- `bo_`: born in the later period;
- `lo_`: lost in the later period.

## Notes for Users

- Start with the notebooks listed above.
- Edit notebook configuration cells for paths, target lemma/POS, periods, and
  output locations.
- Use `Stanza_Data_Prep.ipynb` when starting from raw sentence files; if you
  already have a parsed corpus in SynFlow's seven-field format, start directly
  with the dimension-specific notebooks.
- Choose the Stanza language and model configuration in the notebook before
  parsing raw data.
- Do not edit SynFlow core functions or internal pattern definitions for normal
  use.
- Keep the corpus format consistent with the expected seven-field parsed-token
  layout.
- For large corpora, notebook cells that extract observations may take time;
  increase worker counts only from notebook parameters when available.

## License
This project is licensed under the MIT License.

You are free to use, share, modify, and distribute this software for any purpose, including commercial use, provided that the original copyright notice and permission notice are included. 

**Academic Citation:** If you use this code for academic research, you are requested to cite our paper as detailed in the Citation section below.

[![License: MIT](https://shields.io)](https://opensource.org)


## Citation

If you use SynFlow in academic work, please cite the project and the associated
publication or repository record when available.
