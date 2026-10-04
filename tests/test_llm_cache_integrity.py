import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from incident_routing_bandit import llm_eval
from incident_routing_bandit.synthetic import CATEGORIES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_cache_schema_and_provenance():
    cache = llm_eval.load_llm_cache(DATA_DIR)
    assert "metadata" in cache and "predictions" in cache
    meta = cache["metadata"]
    for key in ("model", "generated_at", "task", "n_incidents", "measured_accuracy"):
        assert key in meta
    assert meta["n_incidents"] == len(cache["predictions"])
    assert len(cache["predictions"]) == 80

    seen_ids = set()
    for p in cache["predictions"]:
        assert p["predicted_category"] in CATEGORIES
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["incident_id"] not in seen_ids
        seen_ids.add(p["incident_id"])


def test_eval_summary_confusion_matrices_are_well_formed():
    summary = llm_eval.load_eval_summary(DATA_DIR)
    for key in ("llm", "keyword"):
        cm = summary[key]["confusion_matrix"]
        assert set(cm.keys()) == set(CATEGORIES)
        for row in cm.values():
            assert set(row.keys()) == set(CATEGORIES)
            assert all(v >= 0 for v in row.values())
        recomputed_accuracy = llm_eval.accuracy_from_confusion_counts(cm)
        assert abs(recomputed_accuracy - summary[key]["accuracy"]) < 1e-9


def test_llm_accuracy_is_plausible_not_hardcoded_perfect():
    summary = llm_eval.load_eval_summary(DATA_DIR)
    acc = summary["llm"]["accuracy"]
    # Must be meaningfully above chance (0.2) and at most 1.0; we do not
    # assert a tight band around the specific measured value so the test
    # does not silently start failing if the cached predictions are
    # regenerated with a different balanced sample.
    assert 0.5 < acc <= 1.0
