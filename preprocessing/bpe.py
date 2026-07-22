"""Learn and apply BPE to tokenized data."""
import tempfile
from pathlib import Path
from subword_nmt import learn_bpe as _learn_bpe, apply_bpe as _apply_bpe
from preprocessing.utils import log, human_size, flatten, atomic_write, short_path
import os

NUM_SYMBOLS = 30000  # Following Kim et al. (2020)


def learn_bpe(input, output):
    """Learn joint BPE from a 'ko\\tja' TSV file."""
    output = Path(output)
    if output.exists():
        log(f"{short_path(output)} already exists ({human_size(output)}), skipping")
        return

    log(f"Loading sentences for BPE from {short_path(input)}...")
    with open(input) as f:
        sentences = flatten(line.strip().split("\t") for line in f)

    log(f"Loaded {len(sentences):,} sentences for BPE. Learning BPE Codes from sentences...")

    with atomic_write(output, suffix=".codes") as f:
        _learn_bpe.learn_bpe(sentences, f, num_symbols=NUM_SYMBOLS)

    log(f"Learned {NUM_SYMBOLS} BPE operations. Saved codes to {short_path(output)} ({human_size(output)})")


def load_bpe(codes_path):
    """Load a BPE model from a codes file."""
    with open(codes_path, encoding="utf-8") as f:
        return _apply_bpe.BPE(f)


def apply_bpe_to_line(line, bpe):
    """Apply BPE to a single ko\\tja line."""
    ko, ja = line.strip().split("\t")
    return f"{bpe.process_line(ko)}\t{bpe.process_line(ja)}"
