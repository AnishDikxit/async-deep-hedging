"""Tests for the C++ market simulator (audit findings C2, B1)."""
import pytest

cpp_exchange = pytest.importorskip("cpp_exchange")

BASE_MU = 0.02
DECAY = 0.98
WARMUP_TICKS = 2_000      # The mean rate relaxes with a time constant of about 121 ticks.
MEASURE_TICKS = 200_000


def mean_event_rate(alpha_jump, seed=7):
    sim = cpp_exchange.MarketSimulator(seed=seed, alpha_jump=alpha_jump)
    for _ in range(WARMUP_TICKS):
        sim.step()
    start = sim.get_event_count()
    for _ in range(MEASURE_TICKS):
        sim.step()
    return (sim.get_event_count() - start) / MEASURE_TICKS


def test_default_process_is_stable():
    sim = cpp_exchange.MarketSimulator()
    assert sim.get_branching_ratio() < 1.0


def test_stable_event_rate_matches_theory():
    # The code decays first and then adds alpha, so n = alpha * decay / (1 - decay).
    alpha = 0.012
    n = alpha * DECAY / (1.0 - DECAY)
    expected = BASE_MU / (1.0 - n)  # 0.0485 events per tick
    # Clustering increases the variance by about 1 / (1 - n)^2, so the SD is about 0.0012.
    assert abs(mean_event_rate(alpha) - expected) < 0.005


def test_old_alpha_locks_in():
    # alpha = 0.15 (n = 7.35) gives an event on almost each tick.
    assert mean_event_rate(0.15) > 0.9


def test_same_seed_gives_same_path():
    a = cpp_exchange.MarketSimulator(seed=123)
    b = cpp_exchange.MarketSimulator(seed=123)
    assert [a.step()[0] for _ in range(5_000)] == [b.step()[0] for _ in range(5_000)]


def test_reset_with_seed_restarts_the_path():
    sim = cpp_exchange.MarketSimulator(seed=5)
    first = [sim.step()[0] for _ in range(1_000)]
    sim.reset(seed=5)
    assert [sim.step()[0] for _ in range(1_000)] == first
