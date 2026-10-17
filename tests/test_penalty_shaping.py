"""Penalty shaping: exact potential-based shaping (math-fixer decision after pilot 3)."""
import numpy as np
import pytest

pytest.importorskip("cpp_exchange")
pytest.importorskip("torch")
from config import Config  # noqa: E402
from environment import VectorTradingEnvironment  # noqa: E402
from train_multicore import apply_cost_shaping, shaping_potential, transform_reward  # noqa: E402

N, T = 4, 200


def shaped_episode(cfg, seeds, actions):
    """Return the raw and shaped discounted returns and Phi(s_0) for one episode."""
    envs = VectorTradingEnvironment(len(seeds), cfg)
    envs.reset(seeds)
    phi0 = shaping_potential(envs, cfg)
    phi_prev, raw_sum, shaped_sum, discount = phi0.copy(), 0.0, 0.0, 1.0
    for t in range(cfg.max_steps):
        _, rewards, done = envs.step(actions[:, t])
        raw = transform_reward(rewards, cfg)
        shaped, phi_prev = apply_cost_shaping(raw, envs, phi_prev, done, cfg)
        raw_sum = raw_sum + discount * raw
        shaped_sum = shaped_sum + discount * shaped
        discount *= cfg.gamma
    return raw_sum, shaped_sum, phi0


@pytest.mark.parametrize("gamma", [0.999, 1.0])
def test_telescoping(gamma):
    cfg = Config(gamma=gamma, max_steps=T)
    actions = np.random.default_rng(3).integers(-1, 2, size=(N, T))
    raw, shaped, phi0 = shaped_episode(cfg, [31, 32, 33, 34], actions)
    np.testing.assert_allclose(shaped, raw - phi0, rtol=0, atol=1e-9)


def test_closed_form_penalty_part():
    # After the terms cancel, step t pays -gamma^49 * penalty(holdings + in_flight at s_{t+1}).
    cfg = Config(gamma=0.999, max_steps=T)
    envs = VectorTradingEnvironment(N, cfg)
    envs.reset([41, 42, 43, 44])
    rng = np.random.default_rng(4)
    for t in range(T):
        phi_p_prev = -envs.projected_penalty(cfg.gamma) / cfg.reward_scale
        penalty_before = envs.total_penalty.copy()
        envs.step(rng.integers(-1, 2, size=N))
        if t + 50 > T:
            break
        raw_penalty = -(envs.total_penalty - penalty_before) / cfg.reward_scale
        phi_p_next = -envs.projected_penalty(cfg.gamma) / cfg.reward_scale
        shaped_penalty = raw_penalty + cfg.gamma * phi_p_next - phi_p_prev
        post_fill = envs.holdings + envs.in_flight_exposure
        excess = np.maximum(0.0, np.abs(post_fill) - cfg.tolerance_band)
        expected = -cfg.gamma ** 49 * cfg.penalty_coef * excess ** 2 / cfg.reward_scale
        np.testing.assert_allclose(shaped_penalty, expected, rtol=0, atol=1e-9)


def test_policy_invariance():
    # Two action sequences on the same seeds: the return difference must not change.
    cfg = Config(gamma=0.999, max_steps=T)
    rng = np.random.default_rng(5)
    a1, a2 = rng.integers(-1, 2, size=(N, T)), rng.integers(-1, 2, size=(N, T))
    seeds = [51, 52, 53, 54]
    raw1, shaped1, _ = shaped_episode(cfg, seeds, a1)
    raw2, shaped2, _ = shaped_episode(cfg, seeds, a2)
    np.testing.assert_allclose(shaped1 - shaped2, raw1 - raw2, rtol=0, atol=1e-9)
