"""Incident routing bandit: contextual multi-armed bandits for on-call alert routing.

This package simulates the problem of routing a newly-raised data-pipeline
incident (an alert) to the on-call track best equipped to resolve it, under
online learning with bandit feedback (we only observe the outcome of the
track we actually chose, never the counterfactual outcome of the others).

The package also contains an offline evaluation of an LLM zero-shot
classifier against a deterministic keyword-matching baseline, used to
derive two empirically measured "context quality" regimes that feed the
bandit simulation.
"""

__version__ = "0.1.0"
