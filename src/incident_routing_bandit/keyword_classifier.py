"""Deterministic rule-based category classifier.

This is the "cheap" baseline context source: a plain substring/keyword
matcher with no learned parameters. It exists so that the LLM zero-shot
classifier evaluated in llm_eval.py has something non-trivial to be
compared against on the exact same evaluation set, rather than against a
hypothetical or borrowed number.
"""
from __future__ import annotations

from typing import Dict, Tuple

from .synthetic import CATEGORIES, KEYWORD_POOL


def _keyword_tokens(category: str) -> list:
    tokens = set()
    for phrase in KEYWORD_POOL[category]:
        for word in phrase.lower().split():
            cleaned = word.strip(".,")
            if len(cleaned) > 3:
                tokens.add(cleaned)
    return sorted(tokens)


_CATEGORY_TOKENS: Dict[str, list] = {c: _keyword_tokens(c) for c in CATEGORIES}


def classify(text: str) -> Tuple[str, float]:
    """Score each category by counting how many of its keyword tokens
    appear as substrings of the incident text, and return the top category
    plus a normalized confidence in [0, 1]. Ties break on the fixed
    CATEGORIES order, so the function is fully deterministic."""
    lowered = text.lower()
    scores = {}
    for category in CATEGORIES:
        scores[category] = sum(1 for tok in _CATEGORY_TOKENS[category] if tok in lowered)
    total = sum(scores.values())
    best_category = CATEGORIES[0]
    best_score = -1
    for category in CATEGORIES:
        if scores[category] > best_score:
            best_score = scores[category]
            best_category = category
    confidence = (best_score / total) if total > 0 else (1.0 / len(CATEGORIES))
    return best_category, confidence


def classify_incident(incident) -> Tuple[str, float]:
    text = f"{incident.title}. {incident.description}"
    return classify(text)
