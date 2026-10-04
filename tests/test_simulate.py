import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from incident_routing_bandit.environment import (
    ContextSource,
    OracleContextSource,
    load_measured_confusion_matrices,
)
from incident_routing_bandit.simulate import run_multi_seed, summarize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
T = 1500
SEEDS = [1, 2, 3, 4, 5, 6]


def _keyword_context_source():
    measured = load_measured_confusion_matrices(DATA_DIR)
    return ContextSource(measured["keyword"], "keyword_matcher")


def test_simulation_is_reproducible_for_a_fixed_seed():
    source = _keyword_context_source()
    r1 = run_multi_seed("lin_ucb", source, T=300, seeds=[42])[0]
    r2 = run_multi_seed("lin_ucb", source, T=300, seeds=[42])[0]
    assert (r1.cumulative_expected_regret == r2.cumulative_expected_regret).all()


def test_context_free_policies_ignore_context_source():
    keyword_source = _keyword_context_source()
    oracle_source = OracleContextSource()
    r1 = run_multi_seed("ucb1", keyword_source, T=300, seeds=[7])[0]
    r2 = run_multi_seed("ucb1", oracle_source, T=300, seeds=[7])[0]
    assert (r1.cumulative_expected_regret == r2.cumulative_expected_regret).all()


def test_contextual_policies_beat_context_free_baselines_with_good_context():
    source = _keyword_context_source()
    lin_ucb = summarize(run_multi_seed("lin_ucb", source, T=T, seeds=SEEDS))
    thompson = summarize(run_multi_seed("thompson_sampling", source, T=T, seeds=SEEDS))
    random_baseline = summarize(run_multi_seed("random", source, T=T, seeds=SEEDS))
    epsilon_greedy = summarize(run_multi_seed("epsilon_greedy", source, T=T, seeds=SEEDS))
    ucb1 = summarize(run_multi_seed("ucb1", source, T=T, seeds=SEEDS))

    # With a near-perfect context (the keyword matcher scores 100% on the
    # 80-incident eval set), contextual policies should accumulate
    # substantially less regret than every context-free baseline. The
    # margin asserted here (20%) is well inside what we actually observe
    # (roughly 45 to 50% less regret), left loose on purpose.
    assert lin_ucb["final_regret_mean"] < 0.8 * random_baseline["final_regret_mean"]
    assert thompson["final_regret_mean"] < 0.8 * random_baseline["final_regret_mean"]
    assert lin_ucb["final_regret_mean"] < 0.8 * epsilon_greedy["final_regret_mean"]
    assert lin_ucb["final_regret_mean"] < 0.8 * ucb1["final_regret_mean"]


def test_cumulative_regret_is_monotonically_nondecreasing():
    source = _keyword_context_source()
    result = run_multi_seed("thompson_sampling", source, T=500, seeds=[11])[0]
    diffs = result.cumulative_expected_regret[1:] - result.cumulative_expected_regret[:-1]
    assert (diffs >= -1e-9).all()


def test_contextual_policy_learns_faster_than_it_explores_in_the_second_half():
    """Per-round average regret in the second half of the run should be
    lower than in the first half: a simple, robust proxy for 'the policy
    is actually learning' that does not depend on exact regret values."""
    source = _keyword_context_source()
    result = run_multi_seed("lin_ucb", source, T=2000, seeds=[3])[0]
    regret = result.cumulative_expected_regret
    first_half_avg = regret[999] / 1000
    second_half_avg = (regret[1999] - regret[999]) / 1000
    assert second_half_avg < first_half_avg
