# Audit corrections for the v2 system (draft)

Source: `docs/AUDIT_REPORT_v2.md`. Full status: `CORRECTIONS_LOG.md`. Verification: `AUDIT_STATE.md`.

## What changed

- C1: The portfolio value includes the initial inventory. The penalty is in the reward only.
- C2: The default `alpha_jump` is 0.012. The event probability is limited to 1.
- C3: Added an MLP agent option and an option that hides E_t.
- C4: Added `evaluate.py` with 1,000 test seeds and two baselines.
- C5: `test.py` reports fills, spread cost and penalty.
- C6: Added `config.py`. The code now uses the README values.
- M1: Comments document the order book limits.
- M2, M6: The README states that the model has no option and that the cores are independent.
- M3: The reward is linear.
- M4: γ is 0.999.
- B1, B6: The C++ engine uses a seeded `std::mt19937`. Each core has its own seeds.
- B2: Rewards are scaled by a running standard deviation.
- B3, B7: Each core runs 16 environments in one batch.
- B4: Removed the unused terminal value.
- B5, B8: Old files are in `legacy/`. `setup.py` builds `src/exchange.cpp`. Added `requirements.txt`.

## Why

- C1: The reported PnL contained the value of the initial shares.
- C2: The branching ratio was 7.35, so the process was unstable.
- C3: No experiment separated E_t from the LSTM.
- C4: One episode with no baseline cannot support a general claim.
- C5: The figure showed many trades, but the trades were not counted.
- C6: The README and the code gave different values.
- M1, M2, M6: The README described features that the code does not have.
- M3: The step utility was almost linear, so it gave no risk aversion.
- M4: γ = 0.99 ignored most of the 1,000-tick episode.
- B1, B6: All workers could start with the same seeds.
- B2: The critic loss controlled the gradient.
- B3, B7: One episode for each update gave high variance and slow training.
- B4, B5, B8: Unused code. A new user could not build the project.

## Deferred

- C1 to C6, M2, M4 to M6: The thesis text is deferred.
- M1: Fills from the order book are a model change. They need a decision and a retrain.

## Blocked

Run these commands from `python/` on the local machine:

1. `python train_multicore.py --run-name main`
2. `python evaluate.py --policies agent rule twap --out results/main`
3. Run the ablation loop in `CORRECTIONS_LOG.md` (20 runs).

The old checkpoints do not load into the new network.

## How to check

1. `pip install -r requirements.txt`
2. `cd python`
3. `python setup.py build_ext --inplace`
4. `cd ..`
5. `python -m pytest`. The expected result is `9 passed`.
6. `cd python`
7. `python evaluate.py --policies rule twap --num-seeds 20`

## Diagrams

- [README diagrams](README.md#8-diagrams)
