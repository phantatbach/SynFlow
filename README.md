# SynFlow

SynFlow is an open-source Python toolkit for multidimensional diachronic
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
use the existing SynFlow functions as they are. Core functions, parser constants,
and corpus patterns are implementation details and should not be edited for
normal use.

## Main Capabilities

SynFlow supports multidimensional diachronic analysis through:

- dependency-based co-occurrence dimensions, such as syntactic slots and their
  lexical fillers;
- constructional configurations extracted from parsed corpora;
- morphological dimensions extracted from CoNLL-U style `FEATS`;
- slot-filler distribution analysis across periods;
- vector-based inspection and incremental clustering of lexical fillers;
- cosine distance, Jensen-Shannon divergence, and total variation distance;
- value-level contribution analysis for interpreting what changed;
- support weighting and permutation testing for more careful historical-corpus
  analysis.

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

Start Jupyter from the repository root:

```bash
jupyter lab
```

The notebooks include the repository path in their setup cells, so no package
installation step is needed beyond installing `requirements.txt`.

Python 3.10 or newer is recommended.

## Requirements

The dependency list is maintained in `requirements.txt`. It includes:

- notebook runtime: `jupyterlab`, `ipykernel`, `ipython`;
- data handling: `pandas`, `numpy`;
- statistics and distance measures: `scipy`, `statsmodels`, `scikit-learn`;
- plotting: `matplotlib`, `plotly`, `seaborn`, `dash`;
- embedding and clustering support: `gensim`, `umap-learn`.

Use this command whenever setting up a new environment:

```bash
python -m pip install -r requirements.txt
```

## Recommended Notebooks

The recommended entry points are in:

```text
case_studies/SynFlow-test/
```

Use these notebooks:

- `SynFlow_Slot_Level.ipynb`
- `SynFlow_Construction_Level.ipynb`
- `SynFlow_Feature_Level.ipynb`
- `SynFlow_SFiller_Level.ipynb`
- `embedding.ipynb`

Other notebooks and lower-level functions are not part of the recommended user
workflow at the moment.

### 1. Slot Level

Use `SynFlow_Slot_Level.ipynb` to inspect syntactic slot distributions around a
target word across periods.

This notebook is for questions such as:

- Which dependency contexts does a target word occur in?
- Which slot types become more or less frequent over time?
- Which periods show stronger changes in slot distribution?

Typical outputs include slot-frequency tables, visualisations of frequent slots,
and distribution-distance results across periods.

### 2. Construction Level

Use `SynFlow_Construction_Level.ipynb` to analyse constructional configurations.

This notebook is for questions such as:

- How does a target construction change across periods?
- Which constructional patterns become more or less frequent?
- How do construction-level distributions compare over time?

It is useful when the relevant unit is not just one dependency relation, but a
larger configuration such as a verb with subject and object branches.

### 3. Feature Level

Use `SynFlow_Feature_Level.ipynb` to analyse morphological features from the
`FEATS` column of parsed corpora.

This notebook is for questions such as:

- Do the morphological properties of a target change over time?
- Which feature types or feature values drive the observed change?
- Are changes concentrated in dimensions such as `Degree`, `Number`, `Case`, or
  `Tense`?

### 4. Slot-Filler Level

Use `SynFlow_SFiller_Level.ipynb` to analyse lexical fillers inside selected
slots.

This notebook is for questions such as:

- Which lexical fillers occur in a target slot?
- Which fillers increase, decrease, appear, or disappear over time?
- Which fillers contribute most to the distance between two periods?

This is the main notebook for value-level interpretation of change.

### 5. Embeddings

Use `embedding.ipynb` to inspect lexical fillers with diachronic embedding
models and clustering.

This notebook is for questions such as:

- How are fillers distributed in embedding space?
- Do fillers form interpretable clusters within a period?
- How do filler clusters develop incrementally across periods?
- Which fillers move between clusters?

Embedding files are not included in this repository. The notebook expects the
user to provide the relevant embedding model files and configure their local
paths inside the notebook.

## Expected Data Layout

The notebooks expect a parsed corpus organised by period or another comparison
group:

```text
corpus_root/
  1995/
    doc_001.txt
    doc_002.txt
  2005/
    doc_001.txt
  2015/
    doc_001.conllu
```

Parsed files should use sentence boundaries like:

```text
<s id=sent-1>
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
functions assume this contract.

## Conceptual Workflow

SynFlow separates representation from temporal analysis:

```text
target occurrences
  -> linguistic observations
  -> period-specific distributions
  -> distance scores
  -> value-level explanations
```

For example:

- slot-level analysis compares distributions over dependency slots;
- construction-level analysis compares distributions over constructional
  configurations;
- feature-level analysis compares distributions over morphological values;
- slot-filler analysis compares distributions over lexical fillers;
- embedding analysis groups fillers into broader thematic clusters.

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
- Do not edit SynFlow core functions, parser constants, or internal pattern
  definitions for normal use.
- Keep the corpus format consistent with the expected seven-field parsed-token
  layout.
- For large corpora, notebook cells that extract observations may take time;
  increase worker counts only from notebook parameters when available.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (CC BY-NC 4.0).

You may use, share, and adapt this code for academic and research purposes with
proper attribution. Commercial use is not allowed.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Citation

If you use SynFlow in academic work, please cite the project and the associated
publication or repository record when available.
