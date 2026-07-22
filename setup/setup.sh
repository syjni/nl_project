#!/bin/bash
# Setup: create venv and install dependencies
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "----------------------------------"
echo "Running Setup..."
echo "----------------------------------"

if ! command -v python3.10 &> /dev/null; then
    echo "python3.10 not found, installing via uv..."
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv python install 3.10
fi

# Find python3.10 (system or uv-installed)
if command -v python3.10 &> /dev/null; then
    PYTHON=python3.10
else
    PYTHON=$(uv python find 3.10)
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR" --prompt cose461-nmt
fi

echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install -q -r "$PROJECT_DIR/requirements.txt"

echo "----------------------------------"
echo "Setup complete!"
echo "----------------------------------"
echo "To activate the environment, run:"
echo "    source .venv/bin/activate"
