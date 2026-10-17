"""Cost shaping must not change the discounted sum of rewards (potential-based shaping)."""
import numpy as np
import pytest

pytest.importorskip("cpp_exchange")
pytest.importorskip("torch")
from config import Config  # noqa: E402
from environment import VectorTradingEnvironment  # noqa: E402
from train_multicore import apply_cost_shaping, transform_reward  # noqa: E402


@pytest.mark.parametrize("gamma", [0.999, 1.0])
def test_shaping_keeps_discounted_return(gamma):
    # Cost part only: its potential is 0 at the start, so the sums must be equal.
    cfg = Config(gamma=gamma, penalty_shaping=False)
    envs = VectorTradingEnvironment(4, cfg)
    envs.reset([21, 22, 23, 24])
    rng = np.random.default_rng(1)
    phi_prev = np.zeros(4)
    raw_sum, shaped_sum, discount, done = np.zeros(4), np.zeros(4), 1.0, False
    while not done:
        _, rewards, done = envs.step(rng.integers(-1, 2, size=4))
        raw = transform_reward(rewards, cfg)
        shaped, phi_prev = apply_cost_shaping(raw, envs, phi_prev, done, cfg)
        raw_sum += discount * raw
        shaped_sum += discount * shaped
        discount *= gamma
    np.testing.assert_allclose(shaped_sum, raw_sum, rtol=0, atol=1e-9)


def test_pending_cost_counts_unfilled_orders():
    envs = VectorTradingEnvironment(3, Config())
    envs.reset([1, 2, 3])
    rng = np.random.default_rng(2)
    done = False
    while not done:
        _, _, done = envs.step(rng.integers(-1, 2, size=3))
    unfilled = envs.orders_placed - envs.num_fills
    np.testing.assert_allclose(envs.pending_cost(), 6.0 * unfilled)
    assert (unfilled > 0).all()  # orders from the last 50 ticks never fill
