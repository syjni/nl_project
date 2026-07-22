"""Download preprocessed data from Google Drive.

Skips if preprocessed data already exists locally.
"""
import zipfile
from pathlib import Path
from preprocessing.utils import log

GDRIVE_ID = "TODO"

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "preprocessed_data"
ZIP_PATH = PROJECT_DIR / "output" / "preprocessed_data.zip"
DONE_MARKER = OUTPUT_DIR / ".download_complete"


def download_preprocessed():
    if DONE_MARKER.exists():
        log("Preprocessed data already downloaded, skipping")
        return

    if GDRIVE_ID == "TODO":
        log("No Google Drive ID set for preprocessed data, skipping download")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        import gdown
        log("Downloading preprocessed data from Google Drive...")
        gdown.download(id=GDRIVE_ID, output=str(ZIP_PATH))
        log(f"Downloaded {ZIP_PATH.stat().st_size / 1e9:.1f} GB")

    log("Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(OUTPUT_DIR.parent)
    ZIP_PATH.unlink()
    DONE_MARKER.touch()
    log("Preprocessed data ready")


if __name__ == "__main__":
    download_preprocessed()
