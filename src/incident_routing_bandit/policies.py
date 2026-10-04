"""Bandit policies.

Two context-free policies (epsilon-greedy, UCB1) and two contextual
policies (LinUCB, linear Thompson Sampling) are implemented from the
primary literature:

- UCB1 follows Auer, Cesa-Bianchi & Fischer (2002), "Finite-time Analysis
  of the Multiarmed Bandit Problem".
- LinUCB (disjoint linear models) follows Li, Chu, Langford & Schapire
  (2010), "A Contextual-Bandit Approach to Personalized News Article
  Recommendation".
- Linear Thompson Sampling follows Agrawal & Goyal (2013), "Thompson
  Sampling for Contextual Bandits with Linear Payoffs".

All policies share the same minimal interface (`select_arm`, `update`) so
that `simulate.py` can run any of them through the identical simulation
loop.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np


class Policy:
    name = "base"

    def select_arm(self, context: Optional[np.ndarray]) -> int:
        raise NotImplementedError

    def update(self, arm: int, context: Optional[np.ndarray], reward: float) -> None:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, n_arms: int, rng: np.random.Generator):
        self.n_arms = n_arms
        self.rng = rng

    def select_arm(self, context: Optional[np.ndarray]) -> int:
        return int(self.rng.integers(0, self.n_arms))

    def update(self, arm: int, context: Optional[np.ndarray], reward: float) -> None:
        pass


class EpsilonGreedyPolicy(Policy):
    name = "epsilon_greedy"

    def __init__(self, n_arms: int, rng: np.random.Generator, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.rng = rng
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def select_arm(self, context: Optional[np.ndarray]) -> int:
        if self.rng.random() < self.epsilon or np.all(self.counts == 0):
            return int(self.rng.integers(0, self.n_arms))
        means = np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.0)
        return int(np.argmax(means))

    def update(self, arm: int, context: Optional[np.ndarray], reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


class UCB1Policy(Policy):
    name = "ucb1"

    def __init__(self, n_arms: int, rng: np.random.Generator):
        self.n_arms = n_arms
        self.rng = rng
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.t = 0

    def select_arm(self, context: Optional[np.ndarray]) -> int:
        self.t += 1
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        means = self.sums / self.counts
        bonus = np.sqrt(2.0 * math.log(self.t) / self.counts)
        return int(np.argmax(means + bonus))

    def update(self, arm: int, context: Optional[np.ndarray], reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


class LinUCBPolicy(Policy):
    """Disjoint linear models, one ridge regression per arm (Li et al. 2010)."""

    name = "lin_ucb"

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.2, ridge_lambda: float = 1.0):
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        self.A = [ridge_lambda * np.eye(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) / ridge_lambda for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta = self.A_inv[a] @ self.b[a]
            mean = theta @ context
            bonus = self.alpha * math.sqrt(max(context @ self.A_inv[a] @ context, 0.0))
            scores[a] = mean + bonus
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        x = context
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        # Sherman-Morrison rank-1 update of the inverse, for speed and
        # numerical parity with the direct inverse (checked in tests).
        Ainv = self.A_inv[arm]
        Ainv_x = Ainv @ x
        denom = 1.0 + x @ Ainv_x
        self.A_inv[arm] = Ainv - np.outer(Ainv_x, Ainv_x) / denom


class LinearThompsonSamplingPolicy(Policy):
    """Linear Thompson Sampling with a Gaussian-Gaussian conjugate model per
    arm (Agrawal & Goyal, 2013)."""

    name = "thompson_sampling"

    def __init__(
        self,
        n_arms: int,
        dim: int,
        rng: np.random.Generator,
        ridge_lambda: float = 1.0,
        noise_sigma: float = 0.5,
    ):
        self.n_arms = n_arms
        self.dim = dim
        self.rng = rng
        self.ridge_lambda = ridge_lambda
        self.noise_sigma = noise_sigma
        self.B = [ridge_lambda * np.eye(dim) for _ in range(n_arms)]
        self.B_inv = [np.eye(dim) / ridge_lambda for _ in range(n_arms)]
        self.f = [np.zeros(dim) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            mu = self.B_inv[a] @ self.f[a]
            cov = (self.noise_sigma ** 2) * self.B_inv[a]
            cov = 0.5 * (cov + cov.T)  # enforce symmetry against float drift
            try:
                theta_sample = self.rng.multivariate_normal(mu, cov)
            except np.linalg.LinAlgError:
                theta_sample = mu
            scores[a] = theta_sample @ context
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        x = context
        self.B[arm] += np.outer(x, x)
        self.f[arm] += reward * x
        Binv = self.B_inv[arm]
        Binv_x = Binv @ x
        denom = 1.0 + x @ Binv_x
        self.B_inv[arm] = Binv - np.outer(Binv_x, Binv_x) / denom


def make_policy(name: str, n_arms: int, dim: int, rng: np.random.Generator) -> Policy:
    if name == "random":
        return RandomPolicy(n_arms, rng)
    if name == "epsilon_greedy":
        return EpsilonGreedyPolicy(n_arms, rng)
    if name == "ucb1":
        return UCB1Policy(n_arms, rng)
    if name == "lin_ucb":
        return LinUCBPolicy(n_arms, dim)
    if name == "thompson_sampling":
        return LinearThompsonSamplingPolicy(n_arms, dim, rng)
    raise ValueError(f"unknown policy name: {name}")


CONTEXT_FREE_POLICIES = {"random", "epsilon_greedy", "ucb1"}
CONTEXTUAL_POLICIES = {"lin_ucb", "thompson_sampling"}
ALL_POLICIES = sorted(CONTEXT_FREE_POLICIES | CONTEXTUAL_POLICIES)
