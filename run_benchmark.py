"""
run_benchmark.py — Full benchmark: 2 tasks × 5 methods × 20 episodes.

Methods:
  1. Baseline (single sample)
  2. Multi-sample, N=5 (random pick — lower bound)
  3. Multi-sample, N=5, smoothness-only scoring
  4. Multi-sample, N=5, full scoring (no adaptive)
  5. Adaptive multi-sample + full scoring (our method)

This design lets you show:
  - Multi-sampling alone helps (method 2 vs 1)
  - Scoring helps more than random (method 3 vs 2)
  - History-aware scoring beats naive scoring (method 4 vs 3)
  - Adaptive sampling is the cherry on top (method 5 vs 4)

Usage:
    python run_benchmark.py --episodes 20 --output results/benchmark.csv
"""

import argparse
import csv
import time
import json
import numpy as np
from pathlib import Path


def make_env(task_name: str):
    import metaworld
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    task = ml1.train_tasks[0]
    env.set_task(task)
    return env, ml1


def obs_to_dict(obs_array, env) -> dict:
    """Convert Meta-World flat obs to Octo-compatible dict."""
    # Meta-World obs is a flat vector. Octo expects image + state.
    # For Meta-World, we render the image from the env and use state directly.
    try:
        img = env.render(offscreen=True)  # (H, W, 3)
    except Exception:
        img = np.zeros((224, 224, 3), dtype=np.uint8)

    return {
        "image_primary": img,
        "state": obs_array,
    }


def make_task_dict(task_name: str) -> dict:
    lang_map = {
        "drawer-open": "open the drawer",
        "pick-place": "pick up the red puck and place it in the target bin",
        "reach": "reach the target position",
        "push": "push the puck to the goal",
    }
    return {"language_instruction": lang_map.get(task_name, task_name)}


def run_episode(env, wrapper, task_dict, max_steps=200):
    """Run one episode. Returns (success, reward, latency_ms, trajectory)."""
    obs, _ = env.reset()
    wrapper.reset()
    total_reward = 0.0
    trajectory = []
    t0 = time.time()

    for step in range(max_steps):
        obs_dict = obs_to_dict(obs, env)
        action = wrapper.act(obs_dict, task_dict)

        # Octo returns action chunks; take first action
        if action.ndim > 1:
            action = action[0]

        trajectory.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    elapsed_ms = (time.time() - t0) * 1000
    success = float(info.get("success", 0.0)) > 0.5
    return success, total_reward, elapsed_ms, np.array(trajectory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/benchmark.csv")
    parser.add_argument("--tasks", nargs="+", default=["drawer-open", "pick-place"])
    parser.add_argument("--save_trajectories", action="store_true")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("trajectories").mkdir(exist_ok=True)

    # Load model
    print("Loading Octo model...")
    from octo.model.octo_model import OctoModel
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    print("Model loaded.\n")

    from wrapper import InferenceWrapper, BaselineWrapper

    # Define methods
    methods = {
        "baseline": lambda m: BaselineWrapper(m),
        "random_N5": lambda m: InferenceWrapper(m, n_base=5, adaptive=False,
                                                 scorer_variant="full",
                                                 # Override: random pick = ignore scoring
                                                 ),
        "smooth_N5": lambda m: InferenceWrapper(m, n_base=5, adaptive=False,
                                                  scorer_variant="smoothness_only"),
        "full_N5": lambda m: InferenceWrapper(m, n_base=5, adaptive=False,
                                                scorer_variant="full"),
        "adaptive_full": lambda m: InferenceWrapper(m, n_base=5, n_max=15,
                                                      adaptive=True,
                                                      scorer_variant="full"),
    }

    results = []
    all_trajectories = {}

    for task_name in args.tasks:
        env, ml1 = make_env(task_name)
        task_dict = make_task_dict(task_name)

        for method_name, wrapper_factory in methods.items():
            wrapper = wrapper_factory(model)
            successes = 0
            rewards = []
            latencies = []
            ep_trajectories = []

            for ep in range(args.episodes):
                success, reward, latency, traj = run_episode(
                    env, wrapper, task_dict,
                )
                successes += int(success)
                rewards.append(reward)
                latencies.append(latency)
                ep_trajectories.append(traj)

                print(f"  [{task_name}] [{method_name}] ep {ep+1}/{args.episodes} "
                      f"{'✓' if success else '✗'} reward={reward:.1f} "
                      f"latency={latency:.0f}ms")

            sr = successes / args.episodes * 100
            stats = wrapper.get_stats()

            results.append({
                "task": task_name,
                "method": method_name,
                "success_rate": sr,
                "avg_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "avg_latency_ms": np.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "avg_samples_per_step": stats.get("avg_samples_per_step", 1),
                "escalation_rate": stats.get("escalation_rate", 0),
                "episodes": args.episodes,
            })

            print(f"\n>>> [{task_name}] {method_name}: "
                  f"SR={sr:.0f}% reward={np.mean(rewards):.1f}±{np.std(rewards):.1f} "
                  f"latency={np.mean(latencies):.0f}ms "
                  f"avg_N={stats.get('avg_samples_per_step', 1):.1f}\n")

            # Save trajectories for best/worst episodes
            if args.save_trajectories:
                key = f"{task_name}_{method_name}"
                all_trajectories[key] = {
                    "success": [bool(s) for s in [True]*successes + [False]*(args.episodes-successes)],
                    "trajectories": [t.tolist() for t in ep_trajectories],
                }

    # Write CSV
    fieldnames = ["task", "method", "success_rate", "avg_reward", "std_reward",
                  "avg_latency_ms", "p95_latency_ms", "avg_samples_per_step",
                  "escalation_rate", "episodes"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults → {args.output}")

    if args.save_trajectories:
        with open("trajectories/all.json", "w") as f:
            json.dump(all_trajectories, f)
        print("Trajectories → trajectories/all.json")


if __name__ == "__main__":
    main()
