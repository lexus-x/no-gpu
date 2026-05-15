"""
test_figures.py — Generate figures with synthetic data to verify plotting pipeline.
Run this BEFORE the real benchmark to make sure figures look right.
Replace with real data after benchmark runs.

Usage:
    python test_figures.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

Path("results").mkdir(exist_ok=True)

np.random.seed(42)

# Plausible synthetic data based on published Octo Meta-World results
# These numbers are CONSERVATIVE estimates, not fabricated peaks
data = [
    # drawer-open
    {"task": "drawer-open", "method": "baseline",       "success_rate": 48, "avg_reward": 38, "std_reward": 12, "avg_latency_ms": 95,  "p95_latency_ms": 115, "avg_samples_per_step": 1.0, "escalation_rate": 0.0,  "episodes": 20},
    {"task": "drawer-open", "method": "random_N5",      "success_rate": 53, "avg_reward": 42, "std_reward": 11, "avg_latency_ms": 445, "p95_latency_ms": 510, "avg_samples_per_step": 5.0, "escalation_rate": 0.0,  "episodes": 20},
    {"task": "drawer-open", "method": "smooth_N5",      "success_rate": 57, "avg_reward": 45, "std_reward": 10, "avg_latency_ms": 450, "p95_latency_ms": 515, "avg_samples_per_step": 5.0, "escalation_rate": 0.0,  "episodes": 20},
    {"task": "drawer-open", "method": "full_N5",        "success_rate": 61, "avg_reward": 49, "std_reward": 9,  "avg_latency_ms": 455, "p95_latency_ms": 520, "avg_samples_per_step": 5.0, "escalation_rate": 0.0,  "episodes": 20},
    {"task": "drawer-open", "method": "adaptive_full",  "success_rate": 65, "avg_reward": 52, "std_reward": 8,  "avg_latency_ms": 520, "p95_latency_ms": 780, "avg_samples_per_step": 6.2, "escalation_rate": 0.35, "episodes": 20},
    # pick-place
    {"task": "pick-place",  "method": "baseline",       "success_rate": 32, "avg_reward": 25, "std_reward": 14, "avg_latency_ms": 98,  "p95_latency_ms": 118, "avg_samples_per_step": 1.0, "escalation_rate": 0.0,  "episodes": 20},
    {"task": "pick-place",  "method": "random_N5",      "success_rate": 36, "avg_reward": 29, "std_reward": 13, "avg_latency_ms": 460, "p95_latency_ms": 530, "avg_samples_per_step": 5.0, "escalation_rate": 0.0,  "episodes": 20},
    {"task": "pick-place",  "method": "smooth_N5",      "success_rate": 40, "avg_reward": 32, "std_reward": 12, "avg_latency_ms": 465, "p95_latency_ms": 535, "avg_samples_per_step": 5.0, "escalation_rate": 0.0,  "episodes": 20},
    {"task": "pick-place",  "method": "full_N5",        "success_rate": 43, "avg_reward": 34, "std_reward": 11, "avg_latency_ms": 470, "p95_latency_ms": 540, "avg_samples_per_step": 5.0, "escalation_rate": 0.0,  "episodes": 20},
    {"task": "pick-place",  "method": "adaptive_full",  "success_rate": 48, "avg_reward": 38, "std_reward": 10, "avg_latency_ms": 540, "p95_latency_ms": 810, "avg_samples_per_step": 6.5, "escalation_rate": 0.40, "episodes": 20},
]

pd.DataFrame(data).to_csv("results/benchmark_synthetic.csv", index=False)

# Scaling curve synthetic data
scaling = [
    {"task": "drawer-open", "n_samples": 1,  "success_rate": 48, "avg_latency_ms": 95,  "p95_latency_ms": 115},
    {"task": "drawer-open", "n_samples": 3,  "success_rate": 56, "avg_latency_ms": 280, "p95_latency_ms": 320},
    {"task": "drawer-open", "n_samples": 5,  "success_rate": 61, "avg_latency_ms": 455, "p95_latency_ms": 520},
    {"task": "drawer-open", "n_samples": 7,  "success_rate": 63, "avg_latency_ms": 630, "p95_latency_ms": 720},
    {"task": "drawer-open", "n_samples": 10, "success_rate": 64, "avg_latency_ms": 890, "p95_latency_ms": 1020},
    {"task": "drawer-open", "n_samples": 15, "success_rate": 65, "avg_latency_ms": 1340,"p95_latency_ms": 1540},
]

pd.DataFrame(scaling).to_csv("results/ablation_scaling_synthetic.csv", index=False)

# Component ablation synthetic data
components = [
    {"task": "drawer-open", "variant": "full",             "n_samples": 5, "success_rate": 61, "avg_latency_ms": 455},
    {"task": "drawer-open", "variant": "no_consistency",   "n_samples": 5, "success_rate": 55, "avg_latency_ms": 450},
    {"task": "drawer-open", "variant": "no_progress",      "n_samples": 5, "success_rate": 58, "avg_latency_ms": 452},
    {"task": "drawer-open", "variant": "consistency_only",  "n_samples": 5, "success_rate": 54, "avg_latency_ms": 448},
    {"task": "drawer-open", "variant": "smoothness_only",   "n_samples": 5, "success_rate": 57, "avg_latency_ms": 450},
]

pd.DataFrame(components).to_csv("results/ablation_components_synthetic.csv", index=False)

print("Synthetic data written to results/*_synthetic.csv")
print("Now run: python generate_figures.py --benchmark results/benchmark_synthetic.csv "
      "--scaling results/ablation_scaling_synthetic.csv "
      "--components results/ablation_components_synthetic.csv")
