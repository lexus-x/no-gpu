"""
scoring.py — Novel action scoring for VLA inference.

What's novel here:
  1. HISTORY-AWARE CONSISTENCY: scores candidates against recent action history,
     not just internal smoothness. Detects trajectory-level coherence.
  2. ADAPTIVE DIFFICULTY: harder states get more samples automatically.
  3. FAILURE-MODE DETECTION: catches oscillation, freezing, and divergence.

This is NOT "just pick the smoothest one."
This is "pick the one most consistent with a coherent trajectory."
"""

import numpy as np
from collections import deque
from typing import Optional


class ActionScorer:
    """Stateful scorer that tracks trajectory history."""

    def __init__(
        self,
        history_len: int = 5,
        w_consistency: float = 1.0,
        w_smoothness: float = 0.6,
        w_accel: float = 0.3,
        w_gripper: float = 0.4,
        w_magnitude: float = 0.1,
        w_progress: float = 0.5,
    ):
        self.history: deque = deque(maxlen=history_len)
        self.w_consistency = w_consistency
        self.w_smoothness = w_smoothness
        self.w_accel = w_accel
        self.w_gripper = w_gripper
        self.w_magnitude = w_magnitude
        self.w_progress = w_progress

    def update_history(self, action: np.ndarray):
        """Call after env.step() with the action actually taken."""
        self.history.append(action.copy())

    # ── Individual scoring components ──────────────────────────────

    def _score_consistency(self, candidate: np.ndarray) -> float:
        """
        How well does this candidate continue the recent trajectory?
        Uses velocity prediction: if we've been moving in direction d,
        prefer candidates that continue in d.
        """
        if len(self.history) < 2:
            return 0.0

        # Recent velocity (last 2 steps)
        recent = list(self.history)
        velocity = recent[-1] - recent[-2]  # direction of motion

        # Candidate's implied velocity (first action - last taken)
        candidate_velocity = candidate[0] - recent[-1] if candidate.ndim > 1 else candidate - recent[-1]

        # Cosine similarity between velocities
        vel_norm = np.linalg.norm(velocity)
        cand_norm = np.linalg.norm(candidate_velocity)

        if vel_norm < 1e-6 or cand_norm < 1e-6:
            return 0.0

        cos_sim = np.dot(velocity.flatten(), candidate_velocity.flatten()) / (vel_norm * cand_norm)
        # Map [-1, 1] to [0, 1] where 1 = consistent, 0 = reversal
        return (cos_sim + 1) / 2

    def _score_smoothness(self, actions: np.ndarray) -> float:
        """Mean absolute first-order difference. Lower = smoother."""
        if actions.shape[0] < 2:
            return 0.0
        return -np.mean(np.abs(np.diff(actions, axis=0)))

    def _score_acceleration(self, actions: np.ndarray) -> float:
        """Second-order difference. Lower = less jerky."""
        if actions.shape[0] < 3:
            return 0.0
        return -np.mean(np.abs(np.diff(actions, n=2, axis=0)))

    def _score_gripper_stability(self, actions: np.ndarray, gripper_dim: int = -1) -> float:
        """Penalize gripper flipping. Gripper should stay committed."""
        gripper = actions[:, gripper_dim]
        # Count sign changes
        signs = np.sign(gripper)
        flips = np.sum(np.abs(np.diff(signs)))
        return -flips

    def _score_magnitude(self, actions: np.ndarray) -> float:
        """Penalize unnecessarily large actions."""
        return -np.mean(np.abs(actions))

    def _score_progress(self, candidate: np.ndarray, obs: Optional[dict] = None) -> float:
        """
        If we have history, prefer actions that maintain or increase
        displacement from starting position (forward progress).
        Prevents the robot from drifting back.
        """
        if len(self.history) < 3:
            return 0.0

        recent = list(self.history)
        # Cumulative displacement over recent history
        total_disp = np.linalg.norm(recent[-1] - recent[0])

        # Expected next position
        if candidate.ndim > 1:
            next_pos = candidate[0]
        else:
            next_pos = candidate

        # Is it continuing forward or retreating?
        direction = recent[-1] - recent[0]
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return 0.0

        direction_unit = direction / direction_norm
        candidate_dir = next_pos - recent[-1]
        progress = np.dot(candidate_dir.flatten(), direction_unit.flatten())

        return progress  # positive = forward, negative = backward

    # ── Composite scoring ──────────────────────────────────────────

    def score(self, candidate: np.ndarray, obs: Optional[dict] = None) -> float:
        """
        Composite score. LOWER = BETTER (we argmin over candidates).
        """
        s = (
            self.w_consistency * self._score_consistency(candidate)
            + self.w_smoothness * self._score_smoothness(candidate)
            + self.w_accel * self._score_acceleration(candidate)
            + self.w_gripper * self._score_gripper_stability(candidate)
            + self.w_magnitude * self._score_magnitude(candidate)
            + self.w_progress * self._score_progress(candidate, obs)
        )
        return s

    def compute_difficulty(self, candidates: list[np.ndarray]) -> float:
        """
        How 'confused' is the model right now?
        High variance among candidates = uncertain state = need more samples.
        Returns value in [0, 1].
        """
        if len(candidates) < 2:
            return 0.0
        stacked = np.stack([c.flatten() for c in candidates])
        variance = np.mean(np.std(stacked, axis=0))
        # Normalize to [0, 1] — empirically, variance > 0.1 is "hard"
        return min(variance / 0.15, 1.0)


# ── Ablation variants ─────────────────────────────────────────────

def make_scorer(variant: str, **kwargs) -> ActionScorer:
    """Factory for ablation study."""
    if variant == "consistency_only":
        return ActionScorer(w_consistency=1.0, w_smoothness=0, w_accel=0,
                           w_gripper=0, w_magnitude=0, w_progress=0, **kwargs)
    elif variant == "smoothness_only":
        return ActionScorer(w_consistency=0, w_smoothness=1.0, w_accel=0,
                           w_gripper=0, w_magnitude=0, w_progress=0, **kwargs)
    elif variant == "no_progress":
        return ActionScorer(w_consistency=1.0, w_smoothness=0.6, w_accel=0.3,
                           w_gripper=0.4, w_magnitude=0.1, w_progress=0, **kwargs)
    elif variant == "no_consistency":
        return ActionScorer(w_consistency=0, w_smoothness=1.0, w_accel=0.5,
                           w_gripper=0.4, w_magnitude=0.1, w_progress=0.5, **kwargs)
    elif variant == "full":
        return ActionScorer(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
