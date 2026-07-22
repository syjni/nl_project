"""Shared utilities for the preprocessing pipeline."""

import inspect
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"


def _log(msg, color):
    caller = inspect.stack()[2].filename
    prefix = os.path.basename(caller)
    tqdm.write(f"{color}[{prefix}] {msg}{RESET}")


def log(msg):
    """Print a cyan log message prefixed with the caller's filename."""
    _log(msg, CYAN)


def log_error(msg):
    """Print a red log message prefixed with the caller's filename."""
    _log(msg, RED)

def log_file_or_dir_already_present(p: Path):
    assert p.exists()

    last_time_modified = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    log(f"{short_path(p)} already exists ({human_size(p)}, last modified {last_time_modified}), skipping creation...")


@contextmanager
def atomic_write(dest, mode="w", suffix=None):
    """Write to a temp file, move to dest on success. Avoids Google Drive sync issues."""
    if suffix is None:
        suffix = Path(dest).suffix
    f = tempfile.NamedTemporaryFile(mode, suffix=suffix, delete=False)
    try:
        yield f
        f.close()
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(f.name).rename(dest)
    except:
        f.close()
        os.unlink(f.name)
        raise


def flatten(lists):
    """Flatten one level of nesting."""
    return [item for sublist in lists for item in sublist]


def human_size(path):
    """Return human-readable file or directory size."""
    p = Path(path)
    if p.is_dir():
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def short_path(path):
    """Return parent_dir/filename, e.g. 'tokenized/train.tsv'."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def dir_has_contents(path):
    """Return True if path is an existing non-empty directory."""
    return path.exists() and any(path.iterdir())


GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"


def _is_hangul(ch):
    cp = ord(ch)
    return (0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F)


def _is_cjk(ch):
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2FA1F or 0xF900 <= cp <= 0xFAFF)


def preview(path, n=10):
    """Print first n lines of a TSV file with color highlighting.

    KO side: Hangul in green, CJK in orange.
    JA side: CJK characters that also appear in the KO side highlighted in orange.
    """
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                print(line.rstrip("\n"))
                continue
            ko, ja = parts

            # Collect CJK chars from KO side
            ko_cjk = set(ch for ch in ko if _is_cjk(ch))

            # Color KO: Hangul green, CJK orange
            ko_colored = []
            for ch in ko:
                if _is_hangul(ch):
                    ko_colored.append(f"{GREEN}{ch}{RESET}")
                elif _is_cjk(ch):
                    ko_colored.append(f"{ORANGE}{ch}{RESET}")
                else:
                    ko_colored.append(ch)

            # Color JA: CJK chars shared with KO in orange
            ja_colored = []
            for ch in ja:
                if ch in ko_cjk:
                    ja_colored.append(f"{ORANGE}{ch}{RESET}")
                else:
                    ja_colored.append(ch)

            print(f"  KO: {''.join(ko_colored)}")
            print(f"  JA: {''.join(ja_colored)}")
            print()
