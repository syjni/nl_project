# COSE461 Hanja-Augmented NMT

Hanja-augmented neural machine translation from Korean to Japanese, developed for the Korea University COSE461 Natural Language Processing course.

The project compares three Korean-to-Japanese NMT preprocessing pipelines:

1. Control: Korean/Japanese tokenization and BPE.
2. Multiple Hanja candidates: append Hanja candidates to Sino-Korean tokens.
3. Single Hanja replacement: replace Sino-Korean words with Hanja before training.

## Repository Contents

```text
.
|-- run.py
|-- preprocess.py
|-- requirements.txt
|-- preprocessing/
|   |-- bpe.py
|   |-- convert_aihub_patent.py
|   |-- download.py
|   |-- download_aihub_patent.py
|   |-- download_preprocessed.py
|   |-- hanja_embedding.py
|   |-- hanja_lookup.py
|   |-- hanja_replacement.py
|   |-- tokenize.py
|   |-- utils.py
|   `-- verify.py
`-- setup/
    `-- setup.sh
```

## Data Notice

Large datasets, preprocessed corpora, model checkpoints, and generated training outputs are not included in this public repository.

If you use the AI Hub patent parallel corpus, download it directly from the official provider and make sure your use follows the dataset license and redistribution terms.

## Setup

Python 3.10 is required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Unix-like systems, you can also use:

```bash
bash setup/setup.sh
```

## Preprocessing

```bash
python preprocess.py
```

Generated files are written under `input/`, `data/`, and `output/`, which are intentionally ignored by Git.

## Training

```bash
python run.py
```

Training checkpoints and logs are written under `output/` and are intentionally excluded from the repository.

## Notes

The code may require local configuration for MeCab, fairseq, and dataset paths depending on the execution environment.
