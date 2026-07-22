"""Download and extract the AI Hub patent parallel corpus.

Downloads from Google Drive and extracts to input/.
"""
import zipfile
from pathlib import Path
from preprocessing.utils import log

GDRIVE_ID = "TODO"
PROJECT_DIR = Path(__file__).parent.parent
INPUT_DIR = PROJECT_DIR / "input"
ZIP_NAME = "118.한-중,한-일 특허 및 기술과학 분야 병렬 말뭉치 데이터"
ZIP_PATH = INPUT_DIR / f"{ZIP_NAME}.zip"
EXTRACT_DIR = INPUT_DIR / ZIP_NAME


def download():
    if GDRIVE_ID == "TODO":
        raise RuntimeError(
            "AI Hub dataset is not bundled with this repository. "
            "Download it from the official provider and place it under input/."
        )

    if EXTRACT_DIR.exists():
        log(f"Already extracted to input/{ZIP_NAME}/, skipping")
        return

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        import gdown
        log(f"Downloading dataset from Google Drive...")
        gdown.download(id=GDRIVE_ID, output=str(ZIP_PATH))
        log(f"Downloaded {ZIP_PATH.stat().st_size / 1e9:.1f} GB")

    log("Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(INPUT_DIR)
    log(f"Extracted to input/{ZIP_NAME}/")


if __name__ == "__main__":
    download()
