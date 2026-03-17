#!/usr/bin/env bash
set -euo pipefail

# Ensure uv installed by install_extra.sh is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Define the usage function
usage() {
    echo "Usage: $0 [--ngrok]"
    exit 1
}

# Initialize NGROK variable
NGROK=""

# Loop through command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngrok)
            NGROK=true
            shift
            ;;
        *)
            usage
            ;;
    esac
done

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it first with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Run the Python app
if [[ -n $NGROK ]]; then
    uv run python -m rag_chatbot --host localhost & ngrok http 7860
else
    uv run python -m rag_chatbot --host localhost
fi
