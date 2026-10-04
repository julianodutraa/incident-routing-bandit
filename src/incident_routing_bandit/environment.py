"""Simulation environment: reward model and context construction.

The environment draws a "true cause category" per round from a fixed
distribution, derives the hidden "optimal track" for that round through
the CONFUSOR_TRACK mapping in synthetic.py (so even a perfect category
classifier cannot reach zero regret: the bandit still has to learn,
from reward feedback, which arm actually resolves each context fastest),
and returns a Bernoulli reward depending on whether the chosen arm
matches the optimal track.

Context quality is varied through `ContextSource` implementations, each
backed by a confusion matrix P(predicted_category | true_category). Two
of those confusion matrices are *measured*, from the offline classifier
evaluation in llm_eval.py (data/classifier_eval_summary.json): the real
zero-shot LLM classifier and the real deterministic keyword matcher,
both scored on the same 80-incident balanced evaluation set. A third,
"legacy_noisy_tagger", is a synthetic low-accuracy regime (not measured
on real predictions) included purely as an illustrative lower anchor, so
the effect of context quality on regret has a visible range to span;
this is stated explicitly wherever it is used, including in the CLI
summary table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .synthetic import CATEGORIES, CONFUSOR_PROBABILITY, CONFUSOR_TRACK, SEVERITIES, TIME_BUCKETS

TRACKS: List[str] = list(CATEGORIES)  # on-call tracks share the same label set as categories

P_GOOD = 0.85
P_BAD = 0.25


@dataclass
class ContextSpec:
    predicted_category_accuracy_source: str
    confusion_matrix: Dict[str, Dict[str, float]]


def load_measured_confusion_matrices(data_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    with open(data_dir / "classifier_eval_summary.json") as f:
        summary = json.load(f)
    out = {}
    for key in ("llm", "keyword"):
        raw = summary[key]["confusion_matrix"]
        out[key] = _normalize_confusion_matrix(raw)
    return out


def _normalize_confusion_matrix(raw_counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    normalized = {}
    for true_cat, row in raw_counts.items():
        total = sum(row.values())
        if total == 0:
            # Laplace fallback: uniform.
            normalized[true_cat] = {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}
        else:
            normalized[true_cat] = {c: (row.get(c, 0) + 0.5) / (total + 0.5 * len(CATEGORIES)) for c in CATEGORIES}
    return normalized


def legacy_noisy_confusion_matrix(target_accuracy: float = 0.5, seed: int = 7) -> Dict[str, Dict[str, float]]:
    """A synthetic confusion matrix for a hypothetical low-quality legacy
    tagger: with probability `target_accuracy` it returns the true
    category, otherwise it returns one of the other four categories
    uniformly at random. This is NOT measured from real predictions; it
    exists only to give the context-quality ablation a clearly worse
    anchor point than the two real classifiers."""
    rng = np.random.default_rng(seed)
    cm = {}
    for true_cat in CATEGORIES:
        row = {c: (1.0 - target_accuracy) / (len(CATEGORIES) - 1) for c in CATEGORIES}
        row[true_cat] = target_accuracy
        cm[true_cat] = row
    return cm


class ContextSource:
    """Draws a predicted-category label for a round, given the true
    category, according to a fixed confusion matrix, then builds the full
    context feature vector (predicted-category one-hot + structured
    nuisance features + bias term)."""

    name = "context_source"
    DIM = 1 + len(CATEGORIES) + len(SEVERITIES) + len(TIME_BUCKETS) + 2  # bias + cat + sev + time + 2 noise dims

    def __init__(self, confusion_matrix: Dict[str, Dict[str, float]], name: str):
        self.confusion_matrix = confusion_matrix
        self.name = name

    def draw_predicted_category(self, rng: np.random.Generator, true_category: str) -> str:
        row = self.confusion_matrix[true_category]
        probs = np.array([row[c] for c in CATEGORIES])
        probs = probs / probs.sum()
        idx = rng.choice(len(CATEGORIES), p=probs)
        return CATEGORIES[idx]

    def build_context(
        self,
        rng: np.random.Generator,
        predicted_category: str,
        severity: str,
        time_bucket: str,
    ) -> np.ndarray:
        cat_onehot = np.array([1.0 if c == predicted_category else 0.0 for c in CATEGORIES])
        sev_onehot = np.array([1.0 if s == severity else 0.0 for s in SEVERITIES])
        time_onehot = np.array([1.0 if t == time_bucket else 0.0 for t in TIME_BUCKETS])
        noise = rng.normal(0.0, 1.0, size=2)
        return np.concatenate([[1.0], cat_onehot, sev_onehot, time_onehot, noise])


class OracleContextSource(ContextSource):
    """Upper-bound sanity check: the context reveals the true category
    directly (accuracy = 1.0 by construction). Used only in tests, to
    confirm that contextual policies approach the best achievable regret
    when context noise is removed entirely."""

    def __init__(self):
        identity = {c: {c2: (1.0 if c2 == c else 0.0) for c2 in CATEGORIES} for c in CATEGORIES}
        super().__init__(identity, name="oracle")


@dataclass
class RoundOutcome:
    true_category: str
    optimal_track: str
    predicted_category: str
    context: Optional[np.ndarray]
    severity: str
    time_bucket: str


class IncidentStream:
    """Generates one simulated incident per round: draws a true category,
    its (hidden) optimal track via the fixed confusor mapping, and
    structured nuisance attributes."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def next_round(self) -> RoundOutcome:
        true_category = CATEGORIES[self.rng.integers(0, len(CATEGORIES))]
        if self.rng.random() < CONFUSOR_PROBABILITY:
            optimal_track = CONFUSOR_TRACK[true_category]
        else:
            optimal_track = true_category
        severity = SEVERITIES[self.rng.integers(0, len(SEVERITIES))]
        time_bucket = TIME_BUCKETS[self.rng.integers(0, len(TIME_BUCKETS))]
        return RoundOutcome(
            true_category=true_category,
            optimal_track=optimal_track,
            predicted_category="",  # filled in by the caller via a ContextSource
            context=None,
            severity=severity,
            time_bucket=time_bucket,
        )


def draw_reward(rng: np.random.Generator, chosen_track: str, optimal_track: str) -> float:
    p = P_GOOD if chosen_track == optimal_track else P_BAD
    return float(rng.random() < p)


def expected_reward(chosen_track: str, optimal_track: str) -> float:
    return P_GOOD if chosen_track == optimal_track else P_BAD
