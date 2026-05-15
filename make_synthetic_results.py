"""
make_synthetic_results.py — Generate realistic synthetic results
for figure/flow testing when you don't have GPU access yet.

Uses plausible numbers based on published Octo Meta-World results.
Replace with real data once benchmark runs.
"""

import csv
import numpy as np

np.random.seed(42)

# Realistic baselines (Octo-base-1.5 on ML1, approximate)
BASELINES = {
    "drawer-open": 52,
    "pick-place": 38,
}

configs = [
    ("baseline", 1),
    ("multisample", 3),
    ("multisample+filter", 3),
    ("multisample", 5),
    ("multisample+filter", 5),
    ("multisample", 10),
    ("multisample+filter", 10),
]

# Gains relative to baseline (approximate, from inference-scaling intuition)
GAINS = {
    ("baseline", 1): 0,
    ("multisample", 3): 5,
    ("multisample+filter", 3): 8,
    ("multisample", 5): 9,
    ("multisample+filter", 5): 13,  # sweet spot
    ("multisample", 10): 12,
    ("multisample+filter", 10): 15,
}

# Latency multipliers
LATENCY_MULT = {
    1: 1.0,
    3: 2.8,
    5: 4.7,
    10: 9.2,
}

results = []
for task, base_sr in BASELINES.items():
    base_latency = np.random.uniform(85, 115)  # ~100ms for single sample
    for method, n in configs:
        gain = GAINS[(method, n)]
        sr = min(base_sr + gain + np.random.normal(0, 1.5), 95)
        latency = base_latency * LATENCY_MULT[n] + np.random.normal(0, 5)
        reward = sr * 0.8 + np.random.normal(0, 2)
        results.append({
            "task": task,
            "method": method,
            "n_samples": n,
            "success_rate": round(sr, 1),
            "avg_reward": round(reward, 1),
            "avg_latency_ms": round(latency, 0),
            "p95_latency_ms": round(latency * 1.15, 0),
            "episodes": 20,
        })

fieldnames = ["task", "method", "n_samples", "success_rate",
              "avg_reward", "avg_latency_ms", "p95_latency_ms", "episodes"]

with open("results/benchmark_synthetic.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("Wrote results/benchmark_synthetic.csv")
print("\nSynthetic results (for testing figures):")
for r in results:
    print(f"  {r['task']:15s} {r['method']:25s} N={r['n_samples']:2d} "
          f"SR={r['success_rate']:5.1f}% latency={r['avg_latency_ms']:.0f}ms")
