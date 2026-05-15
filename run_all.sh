#!/bin/bash
# run_all.sh — One command to get all results + figures.
#
# Usage:
#   bash run_all.sh              # full run (2 tasks, 20 eps)
#   bash run_all.sh --quick      # fast run (1 task, 10 eps)

set -e
cd "$(dirname "$0")"

QUICK=""
if [ "$1" = "--quick" ]; then
    QUICK="--episodes 10 --tasks drawer-open"
    echo "=== QUICK MODE ==="
fi

echo "=== Step 1: Benchmark ==="
python3 run_benchmark.py $QUICK --save_trajectories --output results/benchmark.csv

echo ""
echo "=== Step 2: Ablation (scaling curve) ==="
python3 run_ablation.py --episodes ${QUICK:+10} --task drawer-open

echo ""
echo "=== Step 3: Generate figures ==="
python3 generate_figures.py

echo ""
echo "=== DONE ==="
echo "Figures in figures/"
echo "Tables in results/"
echo ""
ls -la figures/ results/*.md
