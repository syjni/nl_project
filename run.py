"""Train three fairseq Transformer models and evaluate BLEU scores.

Models:
  1. Control                 — no Hanja augmentation
  2. Multiple Hanja          — Hanja candidates concatenated inline
  3. Single Hanja            — Sino-Korean words replaced with Hanja (Kim et al. 2020)
"""
import sys
if sys.version_info[:2] != (3, 10):
    sys.exit(f"Python 3.10 required (got {sys.version_info[0]}.{sys.version_info[1]})")

import subprocess
from pathlib import Path
from preprocessing.download_preprocessed import download_preprocessed
from preprocess import path
from preprocessing.utils import log, dir_has_contents, preview

MODELS_DIR = Path(__file__).parent / "output" / "models"

MODELS = {
    "control": path.control.binarized,
    "multiple_hanja": path.multiple_hanja.binarized,
    "single_hanja": path.single_hanja.binarized,
}

TRAIN_ARGS = [

    # Following Kim et al. (2020) Table 7 (https://aclanthology.org/2020.wat-1.15.pdf)
    "--arch", "transformer",
    "--share-all-embeddings",
    "--encoder-embed-dim", "512",
    "--decoder-embed-dim", "512",
    "--encoder-attention-heads", "8",
    "--decoder-attention-heads", "8",
    "--encoder-layers", "6",
    "--decoder-layers", "6",
    "--optimizer", "adam",
    "--adam-betas", "(0.9, 0.98)",
    "--lr", "0.0005",
    "--lr-scheduler", "inverse_sqrt",
    "--warmup-updates", "4000",
    "--dropout", "0.1",
    "--label-smoothing", "0.1",
    "--max-tokens", "4098",
    "--criterion", "label_smoothed_cross_entropy",

    # TODO: The past paper doens't specify what to put for these.
    # You might want to adjust these :)
    "--max-epoch", "30",
    "--patience", "5",
    "--no-epoch-checkpoints",
    "--best-checkpoint-metric", "bleu",
    "--maximize-best-checkpoint-metric",
    "--eval-bleu",
    "--eval-bleu-args", '{"beam": 5}',
    "--eval-bleu-print-samples",
    "--log-interval", "100",
    "--cpu",  # TODO: <-- especially this probably needs adjusting for fast training...
]


# TODO:
# 1. Update train so that if the model is 'multiple_hanja' we use an attention mask that 
#    ensures hanja candidates cannot attend to each other (mimicking HanjaBridge's methodology)
#    (if too hard, maybe not critical)
#    
# 2. Train the actual models (may take a lot of time!). Will likely need to do this
#    using cloud computing. Make sure you know how you are going to get the model outputs
#    down from the cloud back to your local machine to save them (if they are small enough,
#    maybe we can just use git?)
#    
# 3. Evaluate the models
#    
#    
# Feel free to change anything :)
def train(name, data_dir):
    """Train a single model. Resumes from checkpoint_last.pt if present."""
    save_dir = MODELS_DIR / name

    # Check we have preprocessed data
    if not dir_has_contents(data_dir):
        log(f"{name}: binarized data not found at {data_dir}, run preprocess.py first")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    log(f"Training {name}...")

    # Train the model
    cmd = [
        sys.executable, "-m", "fairseq_cli.train",
        str(data_dir),
        "--save-dir", str(save_dir),
        *TRAIN_ARGS,
    ]
    subprocess.run(cmd, check=True)
    log(f"{name}: training complete")


def main():
    download_preprocessed()

    log("=== Preview ===")
    for name in MODELS:
        log(f"--- {name} ---")
        preview(getattr(path, name).before_binarization.train)
    for name, data_dir in MODELS.items():
        train(name, data_dir)


if __name__ == "__main__":
    main()
