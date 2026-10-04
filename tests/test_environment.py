import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from incident_routing_bandit.environment import P_BAD, P_GOOD, draw_reward


def test_reward_frequencies_match_bernoulli_parameters():
    rng = np.random.default_rng(123)
    n = 20_000
    good_draws = [draw_reward(rng, "db_capacity", "db_capacity") for _ in range(n)]
    bad_draws = [draw_reward(rng, "db_capacity", "k8s_infra") for _ in range(n)]
    good_rate = sum(good_draws) / n
    bad_rate = sum(bad_draws) / n
    assert abs(good_rate - P_GOOD) < 0.02
    assert abs(bad_rate - P_BAD) < 0.02
