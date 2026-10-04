import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from incident_routing_bandit.policies import LinUCBPolicy


def test_sherman_morrison_matches_direct_inverse():
    """LinUCBPolicy maintains A_inv incrementally via the Sherman-Morrison
    rank-1 update instead of recomputing np.linalg.inv(A) every round (an
    actual performance bug avoided, not a toy test: a naive
    O(d^3)-per-round reimplementation would be noticeably slower at
    d ~= 15 and T in the thousands). This test pins the incremental
    inverse against the direct inverse after several updates, catching any
    future refactor that breaks the identity."""
    rng = np.random.default_rng(0)
    dim = 6
    policy = LinUCBPolicy(n_arms=1, dim=dim, alpha=1.0, ridge_lambda=1.0)
    direct_A = np.eye(dim)

    for _ in range(25):
        x = rng.normal(size=dim)
        reward = float(rng.random() < 0.5)
        policy.update(0, x, reward)
        direct_A = direct_A + np.outer(x, x)

    direct_inv = np.linalg.inv(direct_A)
    assert np.allclose(policy.A_inv[0], direct_inv, atol=1e-6)
