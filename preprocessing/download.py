"""Download TALPCo corpus and split into train/dev/test."""
import random
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/matbahasa/TALPCo.git"
DATA_DIR = Path(__file__).parent.parent / "data"
TALPCO_DIR = DATA_DIR / "TALPCo"

SEED = 101
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1


def download():
    """Clone TALPCo repo if not already present."""
    if TALPCO_DIR.exists():
        print("TALPCo already downloaded.")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_URL], cwd=DATA_DIR, check=True)
    print("Downloaded TALPCo.")


def load_lang(path):
    """
    Load sentences for a single language from the dataset.

    Assumes data is stored in the following format:
        [id]\t[sentence] -> i.e. "3\tThe cat sat on the mat"

    Returns a dict of id->sentence.
    """
    sents = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            sents[parts[0]] = parts[1]
    return sents


def split():
    """
    Load Japanese-Korean parallel sentences from TALPCo corpus,
    align by sentence ID, and split into train/dev/test.

    Outputs TSV files with format: Korean\tJapanese
    """
    if (DATA_DIR / "train.tsv").exists():
        print("Already split.")
        return
    
    ja_sents = load_lang(TALPCO_DIR / "jpn" / "data_jpn.txt")
    ko_sents = load_lang(TALPCO_DIR / "kor" / "data_kor.txt")
    
    # Align by sentence ID
    pairs = [(ko_sents[sid], ja_sents[sid]) for sid in ja_sents if sid in ko_sents]
    
    random.seed(SEED)
    random.shuffle(pairs)
    
    n = len(pairs)
    train_end = int(n * TRAIN_RATIO)
    dev_end = train_end + int(n * DEV_RATIO)
    
    splits = {
        "train": pairs[:train_end],
        "dev": pairs[train_end:dev_end],
        "test": pairs[dev_end:],
    }
    
    for name, data in splits.items():
        path = DATA_DIR / f"{name}.tsv"
        path.write_text("\n".join(f"{ko}\t{ja}" for ko, ja in data) + "\n")
        print(f"{name}: {len(data)} pairs -> {path}")

