"""Convert AI Hub patent parallel corpus into ko\tja TSV splits.

Reads labeled data zip files directly from the input directory structure,
extracts ko-ja parallel sentences, and writes train/dev/test TSV files.

The original training data is split into train + dev (100K held out).
AI Hub's validation set becomes our test set (they withhold their own test set).

Use the 'create_splits' function.
"""
import json
import random
import zipfile
from pathlib import Path
from preprocessing.utils import log, atomic_write

SEED = 42
DEV_SIZE = 100_000

PROJECT_DIR = Path(__file__).parent.parent
INPUT_DIR = PROJECT_DIR / "input" / "118.한-중,한-일 특허 및 기술과학 분야 병렬 말뭉치 데이터" / "3.개방데이터" / "1.데이터"
OUTPUT_DIR = PROJECT_DIR / "output" / "preprocessed_data" / "raw_data"


def load_pairs_from_zips(label_dir):
    """Load ko\\tja pairs from all koja/jako zip files in a label directory."""
    pairs = []
    for zip_path in sorted(label_dir.glob("*.zip")):
        if not ("_koja" in zip_path.name or "_jako" in zip_path.name):
            continue
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(name).decode("utf-8-sig"))["data"]
                for entry in data:
                    src_lang = entry["src_lang_code"]
                    for sent in entry["corpus"]:
                        source = sent["source_origin"].strip()
                        translation = sent["translation_final"].strip()
                        if not source or not translation:
                            continue
                        if "\t" in source or "\t" in translation:
                            continue
                        # Normalize to ko\tja regardless of direction
                        if src_lang == "ko":
                            pairs.append((source, translation))
                        else:
                            pairs.append((translation, source))
    return pairs


def write_tsv(pairs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    newline_count = 0
    with atomic_write(path) as f:
        for ko, ja in pairs:
            if "\n" in ko or "\n" in ja or "\r" in ko or "\r" in ja:
                ko = ko.replace("\n", " ").replace("\r", "")
                ja = ja.replace("\n", " ").replace("\r", "")
                newline_count += 1
            f.write(f"{ko}\t{ja}\n")
    msg = f"Wrote {len(pairs):,} pairs to {path.name}"
    if newline_count:
        msg += f" ({newline_count} had embedded newlines)"
    log(msg)


def create_splits():
    expected = [OUTPUT_DIR / f"{s}.tsv" for s in ("train", "dev", "test")]
    if all(f.exists() for f in expected):
        log(f"Splits already exist in {OUTPUT_DIR.name}/, skipping")
        return

    train_pairs = load_pairs_from_zips(INPUT_DIR / "Training" / "02.라벨링데이터")
    log(f"Loaded {len(train_pairs):,} training pairs")

    random.seed(SEED)
    random.shuffle(train_pairs)
    dev_pairs = train_pairs[:DEV_SIZE]
    train_pairs = train_pairs[DEV_SIZE:]

    # AI Hub withholds their test set, so use their validation as ours
    test_pairs = load_pairs_from_zips(INPUT_DIR / "Validation" / "02.라벨링데이터")
    log(f"Loaded {len(test_pairs):,} test pairs (from AI Hub validation)")

    write_tsv(train_pairs, OUTPUT_DIR / "train.tsv")
    write_tsv(dev_pairs, OUTPUT_DIR / "dev.tsv")
    write_tsv(test_pairs, OUTPUT_DIR / "test.tsv")

