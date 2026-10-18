"""The batched environment must give the same episodes as the scalar environment."""
import numpy as np
import pytest

pytest.importorskip("cpp_exchange")
from config import Config  # noqa: E402
from environment import TradingEnvironment, VectorTradingEnvironment  # noqa: E402


@pytest.mark.parametrize("observe_in_flight", [True, False])
def test_vector_env_matches_scalar_env(observe_in_flight):
    cfg = Config(observe_in_flight=observe_in_flight)
    seeds = [11, 12, 13, 14, 15]
    scalar = [TradingEnvironment(cfg) for _ in seeds]
    vector = VectorTradingEnvironment(len(seeds), cfg)

    s_states = np.stack([env.reset(seed=s) for env, s in zip(scalar, seeds, strict=True)])
    v_states = vector.reset(seeds)
    np.testing.assert_array_equal(s_states, v_states)

    rng = np.random.default_rng(0)
    done = False
    while not done:
        actions = rng.integers(-1, 2, size=len(seeds))
        results = [env.step(int(a)) for env, a in zip(scalar, actions, strict=True)]
        v_states, v_rewards, done = vector.step(actions)
        np.testing.assert_array_equal(np.stack([r[0] for r in results]), v_states)
        np.testing.assert_allclose([r[1] for r in results], v_rewards, rtol=0, atol=1e-6)
        assert done == results[0][2]

    np.testing.assert_allclose([e.implementation_shortfall() for e in scalar], vector.implementation_shortfall(), atol=1e-6)
    np.testing.assert_allclose([e.total_penalty for e in scalar], vector.total_penalty, atol=1e-6)
    assert [e.num_fills for e in scalar] == vector.num_fills.tolist()
