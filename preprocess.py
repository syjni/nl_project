"""Download data and run all preprocessing steps.

Three pipelines are run:
  Control:                                  sentences → tokenize → BPE → binarize
  Single Hanja Inline Replacement:          sentences → hanja replace → tokenize → BPE → binarize
  Multiple Hanja Candidate Concatonation:   sentences → tokenize → hanja concat → BPE → binarize
"""
import sys
if sys.version_info[:2] != (3, 10):
    sys.exit(f"Python 3.10 required (got {sys.version_info[0]}.{sys.version_info[1]}). "
             f"Install with: python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt")
import subprocess
import tempfile
import opencc
from preprocessing.verify import verify_has_same_hangul
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
from preprocessing.download_aihub_patent import download as download_aihub_patent
from preprocessing.convert_aihub_patent import create_splits
from preprocessing.tokenize import tokenize_ja, tokenize_ko, strip_pos
from preprocessing.hanja_embedding import append_hanja as _append_hanja
from preprocessing.hanja_replacement import replace_hanja_in_file
from preprocessing.hanja_lookup import get_lookup
from preprocessing.bpe import learn_bpe, load_bpe, apply_bpe_to_line
from preprocessing.utils import log, human_size, atomic_write, log_file_or_dir_already_present, short_path, dir_has_contents, preview

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output" / "preprocessed_data"
TMP_DIR = Path(tempfile.mkdtemp())

MAX_TOKENS = 200  # Following Kim et al. (2020)


class Splits:
    """Paths for a various splits of data (train/dev/test TSVs in one directory)."""
    def __init__(self, dir):
        self.train = dir / "train.tsv"
        self.dev = dir / "dev.tsv"
        self.test = dir / "test.tsv"
        self.bpe_codes = dir / "bpe.codes"
        self.dir = dir
   
    def exists(self):
       return self.train.exists() and self.dev.exists() and self.test.exists()


path = SimpleNamespace(
    raw=Splits(OUTPUT_DIR / "raw_data"),
    raw_tokenized=Splits(OUTPUT_DIR / "raw_tokenized"),
    tmp=Splits(TMP_DIR),
    control=SimpleNamespace(
        before_binarization=Splits(OUTPUT_DIR / "control" / "before_binarization"),
        binarized=OUTPUT_DIR / "control" / "binarized",
    ),
    multiple_hanja=SimpleNamespace(
        before_binarization=Splits(OUTPUT_DIR / "multiple_hanja" / "before_binarization"),
        binarized=OUTPUT_DIR / "multiple_hanja" / "binarized",
    ),
    single_hanja=SimpleNamespace(
        before_binarization=Splits(OUTPUT_DIR / "single_hanja" / "before_binarization"),
        binarized=OUTPUT_DIR / "single_hanja" / "binarized",
        untokenized=Splits(OUTPUT_DIR / "single_hanja" / "untokenized"),
    ),
)

def clear_tmp():
    """Remove all files in the tmp directory."""
    for f in TMP_DIR.iterdir():
        f.unlink()

# --- Core functions ---


def apply_function(fn, input, output, parallel=False):
    """Apply fn to each line of input data, writing results to output path."""
    apply_function_to_file(fn, input=input.train, output=output.train, parallel=parallel)
    apply_function_to_file(fn, input=input.dev, output=output.dev, parallel=parallel)
    apply_function_to_file(fn, input=input.test, output=output.test, parallel=parallel)


def apply_function_to_file(fn, input, output, parallel=False):
    """Apply fn to each line of input, write non-None results to output."""
    if not input.exists():
        raise FileNotFoundError(f"{short_path(input)} does not exist")
    if output.exists() and input != output:
        mtime = datetime.fromtimestamp(output.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        log(f"{short_path(output)} already exists ({human_size(output)}, {mtime}), skipping")
        return

    log(f"Applying {fn.__name__} to each line of {short_path(input)} ({human_size(input)}). Writing output to {short_path(output)}...")

    total = sum(1 for _ in open(input))
    output.parent.mkdir(parents=True, exist_ok=True)

    with atomic_write(output) as fout:
        with open(input) as fin:
            skipped = 0
            if parallel:
                with ThreadPoolExecutor(max_workers=100) as pool:
                    results = pool.map(fn, fin)
                    for line in tqdm(results, total=total, desc=input.name):
                        if line is not None:
                            fout.write(line + "\n")
                        else:
                            skipped += 1
            else:
                for line in tqdm(map(fn, fin), total=total, desc=input.name):
                    if line is not None:
                        fout.write(line + "\n")
                    else:
                        skipped += 1
    log(f"Wrote {short_path(output)} ({human_size(output)})"
        + (f", skipped {skipped:,} lines" if skipped else ""))


def apply_bpe(bpe_model):

    def _apply_bpe(line):
        return apply_bpe_to_line(line, bpe_model)

    return _apply_bpe

def append_hanja(lookup):

    def append_hanja_for_line(line):
        ko, ja = line.strip().split("\t")
        return f"{_append_hanja(ko, lookup)}\t{ja}"

    return append_hanja_for_line

def _tokenize_line(line):
    """Tokenize a ko\\tja line, filter by length."""
    ko, ja = line.strip().split("\t")
    ko = ko.replace("\x00", "")
    ja = ja.replace("\x00", "")
    ko_tokens = tokenize_ko(ko)
    ja_tokens = tokenize_ja(ja)
    if len(ko_tokens.split()) > MAX_TOKENS or len(ja_tokens.split()) > MAX_TOKENS:
        return None
    return f"{ko_tokens}\t{ja_tokens}"


def split_tsv(tsv_path, prefix):
    """Split a ko\\tja TSV into separate .ko and .ja files for fairseq."""
    ko_path = Path(f"{prefix}.ko")
    ja_path = Path(f"{prefix}.ja")
    ko_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Splitting {short_path(tsv_path)} -> {prefix.name}.ko, {prefix.name}.ja...")
    with open(tsv_path) as fin, open(ko_path, "w") as fko, open(ja_path, "w") as fja:
        for line in fin:
            ko, ja = line.strip().split("\t")
            fko.write(ko + "\n")
            fja.write(ja + "\n")


def binarize(input: Splits, output: Path):
    """Split split TSVs into temp files, run fairseq-preprocess, then clean up."""
    if dir_has_contents(output):
        log_file_or_dir_already_present(output)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        split_tsv(input.train, tmpdir / "train")
        split_tsv(input.dev, tmpdir / "dev")
        split_tsv(input.test, tmpdir / "test")

        output.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "fairseq_cli.preprocess",
            "--source-lang", "ko", "--target-lang", "ja",
            "--trainpref", str(tmpdir / "train"),
            "--validpref", str(tmpdir / "dev"),
            "--testpref", str(tmpdir / "test"),
            "--destdir", str(output),
            "--joined-dictionary",
            "--workers", "8",
        ]
        log(f"Running fairseq-preprocess...")
        subprocess.run(cmd, check=True)

    log(f"Binarized data written to {short_path(output)} directory ({human_size(output)})")

# TODO remove name; im sure we can get this programatically
def binarize_pipeline(name, pre_binarization, binarized):
    """Decorator that handles exists-checks, binarization, and preview."""
    def decorator(fn):
        def wrapper():
            log(f"{'=' * 40}")
            log(f"Preprocessing data for {name}")
            log(f"{'=' * 40}")
            if not dir_has_contents(binarized):
                if not pre_binarization.exists():
                    fn()
                else:
                    log_file_or_dir_already_present(pre_binarization)
                log("=== Binarize (for fairseq) ===")
                binarize(input=pre_binarization, output=binarized)
            else:
                log_file_or_dir_already_present(binarized)
            log("=== Preview ===")
            preview(pre_binarization.train)
        return wrapper
    return decorator


@binarize_pipeline("control", path.control.before_binarization, path.control.binarized)
def preprocess_for_control_pipeline():
    log("=== Tokenization ===")
    apply_function(_tokenize_line, input=path.raw, output=path.raw_tokenized)

    log("=== Stripping POS tags ===")
    def _strip_pos_line(line):
        ko, ja = line.strip().split("\t")
        return f"{strip_pos(ko)}\t{ja}"
    apply_function(_strip_pos_line, input=path.raw_tokenized, output=path.tmp)

    log("=== Byte-pair encoding ===")
    learn_bpe(input=path.tmp.train, output=path.tmp.bpe_codes)
    bpe_model = load_bpe(path.tmp.bpe_codes)
    apply_function(apply_bpe(bpe_model), input=path.tmp, output=path.control.before_binarization)

    log("=== Verifying output after BPE is valid (assumes no lines were filtered out!) ===")
    verify_has_same_hangul(path.raw, path.control.before_binarization)
    clear_tmp()


@binarize_pipeline("hanja concatenation", path.multiple_hanja.before_binarization, path.multiple_hanja.binarized)
def preprocess_for_multiple_hanja_pipeline():
    log("=== Tokenization ===")
    apply_function(_tokenize_line, input=path.raw, output=path.raw_tokenized)

    log("=== Embedding Hanja ===")
    # Lookup candidates are already normalized to Traditional Chinese (via OpenCC s2t
    # during dictionary build), matching the JA side's jp2t normalization in tokenize_ja.
    lookup = get_lookup()
    apply_function(append_hanja(lookup), input=path.raw_tokenized, output=path.tmp)

    log("=== Byte-pair Encoding ===")
    learn_bpe(input=path.tmp.train, output=path.tmp.bpe_codes)
    bpe_model = load_bpe(path.tmp.bpe_codes)
    apply_function(apply_bpe(bpe_model), input=path.tmp, output=path.multiple_hanja.before_binarization)

    log("=== Verifying output after BPE is valid (assumes no lines were filtered out!) ===")
    verify_has_same_hangul(path.raw, path.multiple_hanja.before_binarization)
    clear_tmp()


@binarize_pipeline("hanja replacement", path.single_hanja.before_binarization, path.single_hanja.binarized)
def preprocess_for_single_hanja_pipeline():
    log("=== Hanja Replacement (via hanjaro API) ===")
    failure_log = path.single_hanja.untokenized.dir / "failures.log"
    replace_hanja_in_file(path.raw.train, path.single_hanja.untokenized.train, failure_log=failure_log)
    replace_hanja_in_file(path.raw.dev, path.single_hanja.untokenized.dev, failure_log=failure_log)
    replace_hanja_in_file(path.raw.test, path.single_hanja.untokenized.test, failure_log=failure_log)

    log("=== Tokenization ===")
    apply_function(_tokenize_line, input=path.single_hanja.untokenized, output=path.tmp)

    log("=== Stripping POS tags ===")
    def _strip_pos_line(line):
        ko, ja = line.strip().split("\t")
        return f"{strip_pos(ko)}\t{ja}"
    apply_function(_strip_pos_line, input=path.tmp, output=path.tmp)

    log("=== Normalizing Hanja to Traditional Chinese (OpenCC) ===")
    _ko_normalizer = opencc.OpenCC("s2t.json")
    def _normalize_ko_hanja(line):
        ko, ja = line.strip().split("\t")
        return f"{_ko_normalizer.convert(ko)}\t{ja}"
    apply_function(_normalize_ko_hanja, input=path.tmp, output=path.tmp)

    log("=== Byte-pair Encoding ===")
    learn_bpe(input=path.tmp.train, output=path.tmp.bpe_codes)
    bpe_model = load_bpe(path.tmp.bpe_codes)
    apply_function(apply_bpe(bpe_model), input=path.tmp, output=path.single_hanja.before_binarization)

    log("=== Verifying output after BPE is valid (assumes no lines were filtered out!) ===")
    verify_has_same_hangul(path.single_hanja.untokenized, path.single_hanja.before_binarization)
    clear_tmp()


def preprocess():
    download_aihub_patent()
    create_splits()
    preprocess_for_control_pipeline()
    preprocess_for_multiple_hanja_pipeline()
    preprocess_for_single_hanja_pipeline()


if __name__ == "__main__":
    preprocess()
