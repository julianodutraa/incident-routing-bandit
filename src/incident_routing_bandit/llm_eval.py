"""Loaders and metrics for the offline classifier evaluation artifacts
checked into data/: the cached LLM zero-shot predictions (with
provenance) and the derived confusion-matrix summary used by the bandit
simulation's context-quality ablation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .synthetic import CATEGORIES


def load_llm_cache(data_dir: Path) -> dict:
    with open(data_dir / "llm_classification_cache.json") as f:
        return json.load(f)


def load_eval_summary(data_dir: Path) -> dict:
    with open(data_dir / "classifier_eval_summary.json") as f:
        return json.load(f)


def confusion_matrix_is_row_stochastic(cm: Dict[str, Dict[str, float]], tol: float = 1e-6) -> bool:
    for row in cm.values():
        total = sum(row.values())
        if abs(total - 1.0) > tol:
            return False
    return True


def accuracy_from_confusion_counts(raw_counts: Dict[str, Dict[str, int]]) -> float:
    correct = 0
    total = 0
    for true_cat, row in raw_counts.items():
        for pred_cat, count in row.items():
            total += count
            if pred_cat == true_cat:
                correct += count
    if total == 0:
        return 0.0
    return correct / total


def expected_calibration_error(predictions: list, n_bins: int = 5) -> float:
    """predictions: list of {"predicted_category", "confidence", true-label
    implicitly encoded as a boolean 'correct' is NOT available here, so
    this helper expects a list of dicts with an explicit 'correct' bool,
    computed by the caller against ground truth."""
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    n = len(predictions)
    if n == 0:
        return 0.0
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = [p for p in predictions if lo <= p["confidence"] < hi or (hi == 1.0 and p["confidence"] == 1.0)]
        if not in_bin:
            continue
        acc = sum(1 for p in in_bin if p["correct"]) / len(in_bin)
        avg_conf = sum(p["confidence"] for p in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(acc - avg_conf)
    return ece
