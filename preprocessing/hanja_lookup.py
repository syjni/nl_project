"""Build a hangul→hanja dictionary from the 우리말샘 (opendict) XML dump."""

import glob
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET

from preprocessing.utils import log, atomic_write


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HANJA_DIR = os.path.join(_PROJECT_ROOT, "output", "hanja_dictionary")

REPO_URL = "https://github.com/spellcheck-ko/korean-dict-nikl.git"
CLONE_DIR = os.path.join(_HANJA_DIR, "korean-dict-nikl")
OPENDICT_DIR = os.path.join(CLONE_DIR, "opendict")
DICT_PATH = os.path.join(_HANJA_DIR, "hanja_lookup.json")


def _is_hangul(ch):
    return 0xAC00 <= ord(ch) <= 0xD7A3


def _is_cjk(ch):
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2FA1F or 0xF900 <= cp <= 0xFAFF)


def _clean_word(text):
    """Keep only Hangul syllables and '/', then split on '/'."""
    cleaned = "".join(ch for ch in text if _is_hangul(ch) or ch == "/")
    return [w for w in cleaned.split("/") if w]


import opencc

_s2t = opencc.OpenCC("s2t.json")


def _clean_hanja(text):
    # Technically there could be slashes here to indicate alternatives,
    # but that happens like 3 times in the dataset.
    raw = "".join(ch for ch in text if _is_cjk(ch))
    return _s2t.convert(raw)


def _human_size(path):
    """Return human-readable file/dir size."""
    if os.path.isfile(path):
        size = os.path.getsize(path)
    else:
        size = sum(
            os.path.getsize(os.path.join(d, f))
            for d, _, files in os.walk(path) for f in files
        )
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def clone():
    """Sparse-clone only the opendict/ directory."""
    if os.path.isdir(OPENDICT_DIR):
        log(f"Korean dictionary already cloned ({_human_size(CLONE_DIR)}), skipping")
        return
    os.makedirs(_HANJA_DIR, exist_ok=True)
    log("Cloning Korean dictionary (opendict only)...")
    subprocess.run([
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
        REPO_URL, CLONE_DIR,
    ], check=True)
    subprocess.run(
        ["git", "sparse-checkout", "set", "opendict"],
        cwd=CLONE_DIR, check=True,
    )
    log(f"Cloned ({_human_size(CLONE_DIR)})")


def _build_from_file(path):
    """Parse a single opendict XML and return {hangul: [hanja_candidates]}."""
    with open(path) as f:
        content = f.read()
    # Some files have stray control characters that break the XML parser
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
    root = ET.fromstring(content)

    dictionary = {}
    for item in root.findall(".//item"):
        word_info = item.find("wordInfo")
        if word_info is None:
            continue

        raw_word = word_info.findtext("word") or ""
        words = _clean_word(raw_word)
        if not words:
            continue

        origins = word_info.findall(".//original_language_info/original_language")
        candidate = _clean_hanja("".join(o.text or "" for o in origins))
        if not candidate:
            continue

        for word in words:
            if word not in dictionary:
                dictionary[word] = [candidate]
            elif candidate not in dictionary[word]:
                dictionary[word].append(candidate)

    return dictionary


def _merge(base, other):
    """Merge other dict into base, deduplicating candidates."""
    for word, candidates in other.items():
        if word not in base:
            base[word] = candidates
        else:
            for c in candidates:
                if c not in base[word]:
                    base[word].append(c)


def build():
    """Parse all opendict XMLs in parallel and return {hangul: [hanja_candidates]}."""
    from multiprocessing import Pool, cpu_count

    clone()
    log("Building hanja lookup from Korean dictionary...")
    xml_files = sorted(glob.glob(os.path.join(OPENDICT_DIR, "*.xml")))

    with Pool(cpu_count()) as pool:
        results = pool.map(_build_from_file, xml_files)

    dictionary = {}
    for partial in results:
        _merge(dictionary, partial)

    log(f"Built hanja lookup: {len(dictionary):,} words")
    return dictionary


def save(dictionary):
    """Save dictionary to JSON."""
    with atomic_write(DICT_PATH, suffix=".json") as f:
        json.dump(dictionary, f, ensure_ascii=False)
    log(f"Saved hanja lookup to {DICT_PATH} ({_human_size(DICT_PATH)})")


def get_lookup():
    """Load the dictionary from disk, or build and save it if it doesn't exist."""
    if os.path.isfile(DICT_PATH):
        log(f"Hanja lookup already exists ({_human_size(DICT_PATH)}), loading")
        with open(DICT_PATH) as f:
            return json.load(f)
    log("No hanja lookup found — creating one from Korean dictionary")
    dictionary = build()
    save(dictionary)
    return dictionary


if __name__ == "__main__":
    d = get_lookup()
    print(f"{len(d):,} words")
    for word in ["경제", "가", "사회", "교육"]:
        if word in d:
            print(f"  {word}: {d[word][:8]}")
