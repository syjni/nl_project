"""Verification functions for preprocessing pipeline stages."""
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time
from preprocessing.utils import log


def _keep_hangul_only(text):
    """Keep only Hangul characters (syllables + jamo)."""
    return re.sub(r'[^\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]', '', text)


def _line_byte_offsets(path, n_chunks):
    """Find byte offsets splitting a file into n_chunks by line count."""
    with open(path, "rb") as f:
        total_lines = sum(1 for _ in f)

    lines_per_chunk = total_lines // n_chunks
    offsets = [0]
    with open(path, "rb") as f:
        line_num = 0
        for _ in range(n_chunks - 1):
            while line_num < lines_per_chunk * len(offsets):
                f.readline()
                line_num += 1
            offsets.append(f.tell())
    offsets.append(path.stat().st_size)
    return offsets, total_lines


def _check_chunk(args):
    """Verify a chunk of lines between two files have the same Hangul content."""
    path1, path2, start1, end1, start2, end2 = args
    errors = []

    f1 = open(path1, "rb")
    f2 = open(path2, "rb")
    f1.seek(start1)
    f2.seek(start2)

    while f1.tell() < end1:
        line1 = f1.readline().decode("utf-8").strip()
        line2 = f2.readline().decode("utf-8").strip()
        if not line1:
            break

        ko1, ja1 = line1.split("\t")
        ko2, ja2 = line2.split("\t")

        if _keep_hangul_only(ko1) != _keep_hangul_only(ko2):
            errors.append(f"KO mismatch at byte {f1.tell()}")
        if _keep_hangul_only(ja1) != _keep_hangul_only(ja2):
            errors.append(f"JA mismatch at byte {f1.tell()}")

        if len(errors) > 20:
            break

    f1.close()
    f2.close()
    return errors


def verify_has_same_hangul(stage1, stage2):
    """Verify that two stages have identical Hangul content per line.

    Splits both files into chunks at the same line boundaries, then
    checks each chunk in parallel.
    """
    n_workers = cpu_count()
    all_errors = []

    for split_name in ("train", "dev", "test"):
        path1 = getattr(stage1, split_name)
        path2 = getattr(stage2, split_name)
        if not path1.exists() or not path2.exists():
            log(f"Skipping {split_name}: file not found")
            continue

        # Use line count to determine chunk boundaries in both files
        offsets1, total1 = _line_byte_offsets(path1, n_workers)
        offsets2, total2 = _line_byte_offsets(path2, n_workers)

        if total1 != total2:
            all_errors.append(f"{split_name}: line count mismatch ({total1} vs {total2})")
            continue

        chunks = [
            (str(path1), str(path2),
             offsets1[i], offsets1[i + 1],
             offsets2[i], offsets2[i + 1])
            for i in range(n_workers)
        ]

        start = time()
        with Pool(n_workers) as pool:
            results = pool.map(_check_chunk, chunks)
        elapsed = time() - start

        errors = [e for batch in results for e in batch]
        if errors:
            all_errors.extend(f"{split_name}: {e}" for e in errors)
        else:
            log(f"{split_name}: OK ({elapsed:.1f}s, {n_workers} workers)")

    if all_errors:
        log(f"FAILED: {len(all_errors)} errors:")
        for e in all_errors[:20]:
            log(f"  {e}")
    else:
        log("All lines have same Hangul content.")
    return len(all_errors) == 0
