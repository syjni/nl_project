"""Replace Sino-Korean words with Hanja using the hanjaro.juntong.or.kr API.

Sends batches of Korean sentences to the API which returns them with
Sino-Korean words replaced by their Hanja equivalents (Kim et al. 2020 style).
"""
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from preprocessing.utils import log, short_path, human_size, atomic_write

HOST = "hanjaro.juntong.or.kr"
URL = f"http://{HOST}/text_translater.aspx"
BATCH_SIZE = 15000
WORKERS = 2

FAILED_LOG = Path(__file__).parent.parent / "output" / "hanja_replacement_failures.log"


def _query_hanjaro(text):
    """Send text to hanjaro API, return with Sino-Korean words replaced by Hanja."""
    session = requests.Session()
    resp = session.get(URL, timeout=30)
    bs = BeautifulSoup(resp.content.decode(), features="html.parser")
    data = {}
    for f in ["__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION", "TextBox3"]:
        el = bs.find(id=f)
        if el:
            data[f] = el.attrs.get("value", el.text)
    data["TextBox1"] = text
    data["ddlLanguage"] = "1"
    data["1"] = "RadioButton1"  # Direct replacement mode (not bracketed)
    data["ImageButton1.x"] = "120"
    data["ImageButton1.y"] = "20"
    for i in range(1, 9):
        data[f"CheckBox{i}"] = "on"
    form = bs.find(id="form1")
    action = form.attrs.get("action", "/text_translater.aspx")
    resp2 = session.post(f"http://{HOST}/{action}", data=data, timeout=30)
    if resp2.status_code != 200:
        return None
    bs2 = BeautifulSoup(resp2.content.decode(), features="html.parser")
    tb2 = bs2.find(id="TextBox2")
    return tb2.text.replace("\r", "").strip() if tb2 else None


def _sanitize(text):
    """Replace characters that trigger ASP.NET request validation or XML parsing errors."""
    # Strip control characters (except normal whitespace) and Private Use Area characters,
    # which ASP.NET's XML parser rejects
    text = "".join(
        ch for ch in text
        if (ord(ch) >= 32 or ch in "\t\n\r")
        and not (0xE000 <= ord(ch) <= 0xF8FF)   # Private Use Area
        and not (0xFFF0 <= ord(ch) <= 0xFFFF)    # Specials block
    )
    # ASP.NET request validation rejects: <[a-z!/?], &#
    # Replacing < and & with fullwidth equivalents covers all cases
    return (text
        .replace("<", "\uff1c").replace(">", "\uff1e")
        .replace("&", "\uff06")
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
    )


def _restore(text):
    """Restore sanitized characters back to original."""
    return (text
        .replace("\uff1c", "<").replace("\uff1e", ">")
        .replace("\uff06", "&")
    )


def _try_query(lines):
    """Send lines to hanjaro, return replaced lines or None."""
    text = "\n".join(_sanitize(line) for line in lines)
    try:
        result = _query_hanjaro(text)
    except Exception:
        return None
    if result is None:
        return None
    out_lines = [_restore(line) for line in result.split("\n")]
    if len(out_lines) != len(lines):
        return None
    return out_lines


def _process_lines(lines, failure_log):
    """Recursively try to convert lines, halving on failure until single lines."""
    result = _try_query(lines)
    if result is not None:
        return result, 0, 1  # (lines, n_failed, n_api_calls)

    if len(lines) == 1:
        # Single line still fails — keep original, log it
        if failure_log:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(failure_log, "a") as f:
                f.write(f"[{ts}] {lines[0]}\n")
        return lines, 1, 1

    mid = len(lines) // 2
    left, left_failed, left_calls = _process_lines(lines[:mid], failure_log)
    right, right_failed, right_calls = _process_lines(lines[mid:], failure_log)
    return left + right, left_failed + right_failed, left_calls + right_calls + 1


def _process_batch(args):
    """Send a batch of lines to hanjaro, return (batch_idx, replaced_lines)."""
    batch_idx, batch_lines, failure_log = args

    # Try full batch with retries
    for attempt in range(3):
        if attempt > 0:
            time.sleep(1)
        result = _try_query(batch_lines)
        if result is not None:
            return (batch_idx, result)

    # Fallback: recursively halve until we isolate failures
    log(f"Batch {batch_idx}: falling back to sub-batches")
    result, n_failed, n_calls = _process_lines(batch_lines, failure_log)
    if n_failed > 0:
        log_error(f"Batch {batch_idx}: {n_failed} lines failed, kept original ({n_calls} API calls)")
    else:
        log(f"Batch {batch_idx}: resolved with {n_calls} API calls")
    return (batch_idx, result)


def replace_hanja_in_file(input_path, output_path, failure_log=None):
    """Replace Sino-Korean words with Hanja for all lines in a ko\\tja TSV file."""
    if not input_path.exists():
        raise FileNotFoundError(f"{short_path(input_path)} does not exist")
    if output_path.exists():
        log(f"{short_path(output_path)} already exists, skipping")
        return

    log(f"Replacing Hanja in {short_path(input_path)} ({human_size(input_path)})...")

    if failure_log:
        Path(failure_log).parent.mkdir(parents=True, exist_ok=True)

    ko_lines = []
    ja_lines = []
    with open(input_path) as f:
        for line in f:
            ko, ja = line.strip().split("\t")
            ko_lines.append(ko)
            ja_lines.append(ja)

    total = len(ko_lines)
    batch_lines = [ko_lines[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    results = [None] * len(batch_lines)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process_batch, (idx, lines, failure_log)): idx for idx, lines in enumerate(batch_lines)}
        for future in tqdm(as_completed(futures), total=len(batch_lines), desc=input_path.name):
            try:
                batch_idx, out_lines = future.result()
                results[batch_idx] = out_lines
            except Exception as e:
                idx = futures[future]
                log(f"Batch {idx} exception: {e}")
                # Keep original
                start = idx * BATCH_SIZE
                results[idx] = ko_lines[start:start + BATCH_SIZE]

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(output_path) as fout:
        for batch_idx, batch_results in enumerate(results):
            offset = batch_idx * BATCH_SIZE
            for i, ko in enumerate(batch_results):
                fout.write(f"{ko}\t{ja_lines[offset + i]}\n")

    log(f"Wrote {short_path(output_path)} ({human_size(output_path)})")
