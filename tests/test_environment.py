"""Tests for the reward and the PnL report (audit C1)."""
import pytest

pytest.importorskip("cpp_exchange")
from environment import TradingEnvironment  # noqa: E402


def test_shortfall_is_zero_at_reset():
    env = TradingEnvironment(seed=3)
    env.reset(seed=3)
    assert env.implementation_shortfall() == 0.0


def test_first_reward_excludes_initial_inventory_value():
    env = TradingEnvironment()
    for seed in range(20):
        env.reset(seed=seed)
        _, reward, _ = env.step(0)
        # Price moves at most 2 per tick, so a hold changes MTM by at most 2 * |holdings|.
        bound = 2 * abs(env.initial_holdings) + 0.001 * max(0, abs(env.initial_holdings) - 30) ** 2
        assert abs(reward) <= bound + 1e-9


def test_penalty_is_not_taken_from_cash():
    env = TradingEnvironment()
    env.reset(seed=1)
    cash_before = env.cash
    for _ in range(40):  # no fill can arrive before tick 50
        env.step(0)
    assert env.cash == cash_before
    assert env.total_penalty > 0 or abs(env.initial_holdings) <= 30


def test_same_seed_gives_same_episode():
    a, b = TradingEnvironment(), TradingEnvironment()
    a.reset(seed=9); b.reset(seed=9)
    assert a.initial_holdings == b.initial_holdings
    assert [a.step(1)[1] for _ in range(200)] == [b.step(1)[1] for _ in range(200)]
