#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
FILES=(
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
)

for f in "${FILES[@]}"; do
  if [ ! -f "${f%.gz}" ]; then
    echo "Downloading $f..."
    curl -fLO "$BASE/$f"
    gunzip "$f"
  else
    echo "Already have ${f%.gz}, skipping."
  fi
done

echo "Done. Data is in $DATA_DIR"
