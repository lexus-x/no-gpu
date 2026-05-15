"""
wrapper.py — Multi-sample inference wrapper with adaptive sampling.

Novel components:
  1. Adaptive N: starts with N_base, escalates if model is uncertain
  2. History-aware scoring: trajectory consistency, not just smoothness
  3. Failure-mode detection: catches oscillation and stuck states
  4. Clean ablation surface: each component can be toggled off
"""

import numpy as np
from typing import Optional
from scoring import ActionScorer, make_scorer


class InferenceWrapper:
    """
    Drop-in replacement for model.sample_actions().

    Usage:
        wrapper = InferenceWrapper(model, n_base=5)
        for step in episode:
            action = wrapper.act(obs, task)
            env.step(action)
    """

    def __init__(
        self,
        model,
        n_base: int = 5,
        n_max: int = 15,
        scorer_variant: str = "full",
        adaptive: bool = True,
        difficulty_threshold: float = 0.6,
        history_len: int = 5,
    ):
        self.model = model
        self.n_base = n_base
        self.n_max = n_max
        self.adaptive = adaptive
        self.difficulty_threshold = difficulty_threshold
        self.scorer = make_scorer(scorer_variant, history_len=history_len)

        # Telemetry
        self.total_samples = 0
        self.total_steps = 0
        self.difficulty_history = []
        self.n_used_history = []

    def act(self, obs: dict, task: dict) -> np.ndarray:
        """
        Select best action. Handles adaptive sampling internally.
        """
        # Phase 1: draw base samples
        candidates = []
        for _ in range(self.n_base):
            a = np.array(self.model.sample_actions(obs, task))
            candidates.append(a)

        n_used = self.n_base

        # Phase 2: adaptive escalation if uncertain
        if self.adaptive:
            difficulty = self.scorer.compute_difficulty(candidates)
            self.difficulty_history.append(difficulty)

            if difficulty > self.difficulty_threshold:
                extra = min(int((difficulty - self.difficulty_threshold) * 20),
                           self.n_max - self.n_base)
                for _ in range(extra):
                    a = np.array(self.model.sample_actions(obs, task))
                    candidates.append(a)
                n_used += extra

        self.n_used_history.append(n_used)
        self.total_samples += n_used
        self.total_steps += 1

        # Phase 3: score and select
        scores = [self.scorer.score(c, obs) for c in candidates]
        best_idx = int(np.argmin(scores))  # lower = better

        best_action = candidates[best_idx]
        self.scorer.update_history(best_action[0] if best_action.ndim > 1 else best_action)

        return best_action

    def reset(self):
        """Call at episode start."""
        self.scorer.history.clear()

    def get_stats(self) -> dict:
        """Return telemetry for the paper."""
        return {
            "total_samples": self.total_samples,
            "total_steps": self.total_steps,
            "avg_samples_per_step": self.total_samples / max(self.total_steps, 1),
            "avg_difficulty": np.mean(self.difficulty_history) if self.difficulty_history else 0,
            "max_difficulty": np.max(self.difficulty_history) if self.difficulty_history else 0,
            "escalation_rate": np.mean([n > self.n_base for n in self.n_used_history]) if self.n_used_history else 0,
        }


class BaselineWrapper:
    """Single-sample baseline for comparison."""

    def __init__(self, model):
        self.model = model
        self.total_samples = 0
        self.total_steps = 0

    def act(self, obs: dict, task: dict) -> np.ndarray:
        self.total_samples += 1
        self.total_steps += 1
        return np.array(self.model.sample_actions(obs, task))

    def reset(self):
        pass

    def get_stats(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "total_steps": self.total_steps,
            "avg_samples_per_step": 1.0,
        }
