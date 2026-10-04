#!/usr/bin/env python3
"""End-to-end reproducible demo.

Runs the offline classifier comparison (LLM zero-shot vs deterministic
keyword matcher, both scored against the same synthetic ground truth),
then runs the contextual-bandit incident-routing simulation across all
five policies and three context-quality regimes, and prints a summary
report. All numbers printed here are produced by this run, not pasted
from a prior one: re-running reproduces them exactly (fixed seeds).

Usage:
    python scripts/run_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from incident_routing_bandit import llm_eval
from incident_routing_bandit.environment import (
    ContextSource,
    load_measured_confusion_matrices,
    legacy_noisy_confusion_matrix,
)
from incident_routing_bandit.policies import ALL_POLICIES
from incident_routing_bandit.simulate import run_multi_seed, summarize

DATA_DIR = ROOT / "data"
T_ROUNDS = 3000
SEEDS = list(range(20))


def print_offline_classifier_section():
    summary = llm_eval.load_eval_summary(DATA_DIR)
    print("=" * 72)
    print("OFFLINE CLASSIFIER EVALUATION (80 balanced synthetic incidents)")
    print("=" * 72)
    print(f"  Zero-shot LLM classifier accuracy:      {summary['llm']['accuracy']:.4f}")
    print(f"  Deterministic keyword matcher accuracy: {summary['keyword']['accuracy']:.4f}")
    cache = llm_eval.load_llm_cache(DATA_DIR)
    print(f"  LLM provenance: {cache['metadata']['model']}, {cache['metadata']['generated_at']}")
    print()


def build_context_sources():
    measured = load_measured_confusion_matrices(DATA_DIR)
    sources = {
        "legacy_noisy_tagger": ContextSource(legacy_noisy_confusion_matrix(0.5), "legacy_noisy_tagger"),
        "keyword_matcher": ContextSource(measured["keyword"], "keyword_matcher"),
        "llm_zero_shot": ContextSource(measured["llm"], "llm_zero_shot"),
    }
    return sources


def run_bandit_section(sources):
    print("=" * 72)
    print(f"CONTEXTUAL BANDIT SIMULATION (T={T_ROUNDS} rounds, {len(SEEDS)} seeds)")
    print("=" * 72)
    rows = []
    for context_name, context_source in sources.items():
        for policy_name in ALL_POLICIES:
            results = run_multi_seed(policy_name, context_source, T_ROUNDS, SEEDS)
            summary = summarize(results)
            rows.append(summary)
            print(
                f"  context={context_name:20s} policy={policy_name:18s} "
                f"final_regret={summary['final_regret_mean']:7.1f} (+/-{summary['final_regret_std']:5.1f})  "
                f"routing_acc_last_quartile={summary['routing_accuracy_last_quartile_mean']:.3f}"
            )
    return rows


def main():
    print_offline_classifier_section()
    sources = build_context_sources()
    rows = run_bandit_section(sources)
    out_path = ROOT / "examples" / "sample_report.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print()
    print(f"Full results written to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
