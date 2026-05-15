"""
run_ablation.py — Component ablation + N-scaling study.

Two ablation axes:
  1. Scoring component ablation (remove one component at a time)
  2. Sample count scaling curve (N=1,3,5,7,10,15)

Produces:
  - results/ablation_components.csv
  - results/ablation_scaling.csv

Usage:
    python run_ablation.py --episodes 15 --task drawer-open
"""

import argparse
import csv
import time
import numpy as np
from pathlib import Path


def run_episode_scoring(env, model, task_dict, scorer, n_samples, max_steps=200):
    """Run with explicit scorer (not via wrapper) for clean ablation."""
    obs, _ = env.reset()
    scorer.history.clear()
    total_reward = 0.0
    t0 = time.time()

    for step in range(max_steps):
        # Render obs
        try:
            img = env.render(offscreen=True)
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        obs_dict = {"image_primary": img, "state": obs}

        # Multi-sample
        candidates = []
        for _ in range(n_samples):
            a = np.array(model.sample_actions(obs_dict, task_dict))
            candidates.append(a)

        # Score
        scores = [scorer.score(c, obs_dict) for c in candidates]
        best_idx = int(np.argmin(scores))
        action = candidates[best_idx]
        if action.ndim > 1:
            action = action[0]

        scorer.update_history(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    elapsed_ms = (time.time() - t0) * 1000
    success = float(info.get("success", 0.0)) > 0.5
    return success, total_reward, elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--task", type=str, default="drawer-open")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    print("Loading model...")
    from octo.model.octo_model import OctoModel
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    print("Done.\n")

    import metaworld
    ml1 = metaworld.ML1(args.task)
    env = ml1.train_classes[args.task]()
    task = ml1.train_tasks[0]
    env.set_task(task)
    task_dict = {"language_instruction": args.task.replace("-", " ")}

    from scoring import make_scorer

    # ── Ablation 1: Component removal ──────────────────────────────
    print("=" * 60)
    print("ABLATION 1: Scoring Component Removal")
    print("=" * 60)

    component_variants = [
        "full",
        "no_consistency",
        "no_progress",
        "consistency_only",
        "smoothness_only",
    ]

    component_results = []
    for variant in component_variants:
        scorer = make_scorer(variant)
        successes = 0
        latencies = []

        for ep in range(args.episodes):
            success, _, latency = run_episode_scoring(
                env, model, task_dict, scorer, n_samples=5,
            )
            successes += int(success)
            latencies.append(latency)
            print(f"  {variant:20s} ep {ep+1}/{args.episodes} {'✓' if success else '✗'}")

        sr = successes / args.episodes * 100
        component_results.append({
            "task": args.task,
            "variant": variant,
            "n_samples": 5,
            "success_rate": sr,
            "avg_latency_ms": np.mean(latencies),
        })
        print(f"  → {variant}: SR={sr:.1f}%\n")

    with open("results/ablation_components.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=component_results[0].keys())
        writer.writeheader()
        writer.writerows(component_results)

    # ── Ablation 2: N-scaling curve ────────────────────────────────
    print("=" * 60)
    print("ABLATION 2: Sample Count Scaling")
    print("=" * 60)

    n_values = [1, 3, 5, 7, 10, 15]
    scaling_results = []

    for n in n_values:
        scorer = make_scorer("full")
        successes = 0
        latencies = []

        for ep in range(args.episodes):
            success, _, latency = run_episode_scoring(
                env, model, task_dict, scorer, n_samples=n,
            )
            successes += int(success)
            latencies.append(latency)

        sr = successes / args.episodes * 100
        scaling_results.append({
            "task": args.task,
            "n_samples": n,
            "success_rate": sr,
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
        })
        print(f"  N={n:2d}: SR={sr:.1f}% latency={np.mean(latencies):.0f}ms")

    with open("results/ablation_scaling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scaling_results[0].keys())
        writer.writeheader()
        writer.writerows(scaling_results)

    print("\nDone. Results in results/ablation_*.csv")


if __name__ == "__main__":
    main()
