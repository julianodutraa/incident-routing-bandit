"""Simulation loop: runs one policy against one context source for T
rounds and returns the per-round and cumulative expected regret.

Expected (rather than realized) regret is accumulated at each round,
which is the standard choice in the bandit literature when the reward
gap is known by construction in a simulation: it removes the sampling
noise of Bernoulli reward draws from the regret curve itself, while the
policy still only ever observes realized Bernoulli rewards and must
learn from those alone. This is made explicit here, and tests pin it
down with a reproducibility check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .environment import ContextSource, IncidentStream, TRACKS, draw_reward, expected_reward
from .policies import CONTEXT_FREE_POLICIES, Policy, make_policy
from .synthetic import CATEGORIES


@dataclass
class SimulationResult:
    policy_name: str
    context_source_name: str
    seed: int
    cumulative_expected_regret: np.ndarray  # shape (T,)
    cumulative_realized_reward: np.ndarray  # shape (T,)
    routing_accuracy_last_quartile: float  # fraction of optimal-arm picks in the last 25% of rounds


def run_single_simulation(
    policy_name: str,
    context_source: ContextSource,
    T: int,
    seed: int,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    stream = IncidentStream(rng)
    is_contextual = policy_name not in CONTEXT_FREE_POLICIES
    dim = ContextSource.DIM if is_contextual else 0
    policy: Policy = make_policy(policy_name, n_arms=len(TRACKS), dim=dim, rng=rng)

    cum_regret = np.zeros(T)
    cum_reward = np.zeros(T)
    optimal_hits = np.zeros(T, dtype=bool)
    running_regret = 0.0
    running_reward = 0.0

    for t in range(T):
        round_outcome = stream.next_round()
        predicted_category = context_source.draw_predicted_category(rng, round_outcome.true_category)
        context_vec: Optional[np.ndarray] = None
        if is_contextual:
            context_vec = context_source.build_context(
                rng, predicted_category, round_outcome.severity, round_outcome.time_bucket
            )
        arm_idx = policy.select_arm(context_vec)
        chosen_track = TRACKS[arm_idx]
        reward = draw_reward(rng, chosen_track, round_outcome.optimal_track)
        policy.update(arm_idx, context_vec, reward)

        best_reward = expected_reward(round_outcome.optimal_track, round_outcome.optimal_track)
        chosen_expected = expected_reward(chosen_track, round_outcome.optimal_track)
        running_regret += best_reward - chosen_expected
        running_reward += reward
        cum_regret[t] = running_regret
        cum_reward[t] = running_reward
        optimal_hits[t] = chosen_track == round_outcome.optimal_track

    last_quartile_start = int(T * 0.75)
    routing_accuracy = float(np.mean(optimal_hits[last_quartile_start:]))

    return SimulationResult(
        policy_name=policy_name,
        context_source_name=context_source.name,
        seed=seed,
        cumulative_expected_regret=cum_regret,
        cumulative_realized_reward=cum_reward,
        routing_accuracy_last_quartile=routing_accuracy,
    )


def run_multi_seed(
    policy_name: str,
    context_source: ContextSource,
    T: int,
    seeds: List[int],
) -> List[SimulationResult]:
    return [run_single_simulation(policy_name, context_source, T, seed) for seed in seeds]


def summarize(results: List[SimulationResult]) -> dict:
    final_regrets = np.array([r.cumulative_expected_regret[-1] for r in results])
    accuracies = np.array([r.routing_accuracy_last_quartile for r in results])
    return {
        "policy": results[0].policy_name,
        "context_source": results[0].context_source_name,
        "n_seeds": len(results),
        "T": len(results[0].cumulative_expected_regret),
        "final_regret_mean": float(final_regrets.mean()),
        "final_regret_std": float(final_regrets.std()),
        "routing_accuracy_last_quartile_mean": float(accuracies.mean()),
        "routing_accuracy_last_quartile_std": float(accuracies.std()),
    }
