#!/bin/bash
# One-shot setup — run on a GPU machine with CUDA
# Tested on Ubuntu 22.04, Python 3.10, CUDA 12.x

set -e

echo "=== Step 1: Clone and install Octo ==="
if [ ! -d "octo" ]; then
    git clone https://github.com/octo-models/octo
fi
cd octo
pip install -e . 2>&1 | tail -5
cd ..

echo "=== Step 2: Install Meta-World + deps ==="
pip install metaworld 2>&1 | tail -3
pip install matplotlib pandas seaborn imageio 2>&1 | tail -3

echo "=== Step 3: Verify ==="
python3 -c "
from octo.model.octo_model import OctoModel
print('Octo import OK')
import metaworld
print('Meta-World import OK')
"

echo "=== DONE ==="
