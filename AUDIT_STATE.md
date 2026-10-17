# Audit State

This file records the verification of each fixed finding.
Each row gives the evidence that we examined after the fix.

| ID | area | status | evidence | verified |
|---|---|---|---|---|
| C1 | code | fixed | `python/environment.py:47` initial MTM; `:100` reward = ΔMTM − penalty; no `cash -= inventory_bleed`; `tests/test_environment.py` passes | verified |
| C1 | README | fixed | `README.md` Results table: ~$19,000 labelled; −$6,400 labelled as audit estimate | verified |
| C2 | code | fixed | `src/exchange.cpp:99` default 0.012; `:139` min(λ, 1); `tests/test_exchange.py` rate 0.0485 ± 0.005 passes; 0.15 gives > 0.9 | verified |
| C3 | code | blocked | `--arch mlp` and `--no-in-flight` run in smoke test; training blocked: local run | verified (code only) |
| C3 | README | fixed | `README.md` Background and Results: MDP with delayed actions, no ablation | verified |
| C4 | code | blocked | `python/evaluate.py:136` test seeds; `python/train_multicore.py:231` winner copy; baselines ran on 20 seeds | verified (code only) |
| C4 | README | fixed | `README.md` "New metrics: pending a local run" | verified |
| C5 | code | fixed | `python/test.py` prints fills, spread cost, penalty (smoke run) | verified |
| C5 | README | fixed | `README.md` Fig. 4.3 text: decaying limit cycle in one episode | verified |
| C6 | config | fixed | `python/config.py:39,41`; `python/agent.py:15`; `python/environment.py:31`; `python/train_multicore.py:150,159` | verified |
| C6 | README | fixed | `README.md` hyperparameter table matches `python/config.py` | verified |
| C6 | comment | fixed | no "3 Features" in `python/*.py` | verified |
| M1 | README | fixed | `README.md` Simulator rules and Known limits | verified |
| M1 | code | deferred | comments in `src/exchange.cpp` only | n/a |
| M2 | README | fixed | `README.md` section 1 and Known limits | verified |
| M3 | code | fixed | `python/config.py:44` linear default | verified |
| M4 | config | fixed | `python/config.py:36` γ = 0.999 | verified |
| M6 | README | fixed | `README.md` section 1; start message in `python/train_multicore.py` | verified |
| B1 | code | fixed | `python/train_multicore.py:59`; `std::mt19937` in `src/exchange.cpp`; same-seed tests pass | verified |
| B2 | code | fixed | `python/train_multicore.py:103`; loss ratio in log (smoke: 0.01) | verified |
| B3 | code | fixed | `num_envs` 16, `num_minibatches` 4 in `python/config.py`; smoke training ran | verified |
| B4 | code | fixed | no `next_state_value` in `python/train_multicore.py` | verified |
| B5 | code | fixed | `src/` contains only `exchange.cpp`; `legacy/main.cpp` | verified |
| B6 | code | fixed | no `rand() %` in `src/exchange.cpp` | verified |
| B7 | code | fixed | batched `select_action` in `python/agent.py` | verified |
| B8 | code | fixed | `python setup.py build_ext --inplace` builds from `python/`; `requirements.txt` | verified |
| B8 | README | fixed | clone URL, no `.pyd`, `legacy/` historical | verified |
| M5 | thesis | deferred | deferred: thesis | n/a |

Lint: `ruff check --isolated --select F,B` passes on changed Python files. `g++ -Wall -Wextra` gives no warnings on `src/exchange.cpp`.
Tests: `python -m pytest` gives 9 passed.
