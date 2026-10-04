import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from incident_routing_bandit import keyword_classifier as kc
from incident_routing_bandit.synthetic import generate_eval_dataset


def test_classify_unambiguous_examples():
    cat, conf = kc.classify("Tablespace usage above threshold on the production volume.")
    assert cat == "db_capacity"
    assert conf > 0.5

    cat, conf = kc.classify("Pod stuck in CrashLoopBackOff after the latest deploy.")
    assert cat == "k8s_infra"


def test_classify_empty_text_falls_back_to_uniform_confidence():
    cat, conf = kc.classify("")
    assert cat in {"db_capacity", "dag_orchestration", "k8s_infra", "data_quality", "network_connectivity"}
    assert abs(conf - 0.2) < 1e-9


def test_accuracy_on_balanced_eval_set_beats_chance_by_a_wide_margin():
    incidents = generate_eval_dataset(n_per_category=16, seed=20261004)
    correct = 0
    for inc in incidents:
        pred_cat, _ = kc.classify_incident(inc)
        if pred_cat == inc.true_category:
            correct += 1
    accuracy = correct / len(incidents)
    # Chance for 5 balanced classes is 0.20. We assert a loose, non-cherry-picked
    # bound (not the exact measured value, which is recorded separately in
    # data/classifier_eval_summary.json) so this test stays meaningful even if
    # the synthetic generator's keyword pools are tweaked later.
    assert accuracy > 0.9
