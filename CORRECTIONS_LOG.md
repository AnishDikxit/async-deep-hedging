# Corrections Log

Source: `docs/AUDIT_REPORT_v2.md` (audit of commit `13391e9`).
Branch: `audit-corrections`, made from `master`.

Status values: todo, in progress, fixed, blocked, deferred.

## Findings

| ID | area | severity | status | files changed | note |
|---|---|---|---|---|---|
| B8 | README | Minor | todo | | Wrong clone URL, missing `requirements.txt`, missing `.pyd`, v1 run steps. |
| C1 | README | Critical | todo | | README PnL includes the initial inventory value. |
| C3 | README | Critical | todo | | README gives the improvement to the LSTM with no ablation. |
| C4 | README | Critical | todo | | README results use one episode and no baseline. |
| C5 | README | Critical | todo | | README text does not agree with Fig. 4.3. |
| C6 | README | Critical | todo | | README hyperparameters do not agree with the code. |
| M1 | README | Major | todo | | README describes liquidity effects that the code does not have. |
| M2 | README | Major | todo | | README says "delta-neutral", but the model has no option. |
| M6 | README | Major | todo | | README says "distributed rollout"; the cores are independent runs. |
| C1 | code | Critical | todo | | Initial inventory not in portfolio value; penalty taken from cash. |
| C2 | code | Critical | todo | | Hawkes branching ratio 7.5; probability not limited to 1. |
| C3 | code | Critical | todo | | No feed-forward agent for an ablation. |
| C4 | code | Critical | todo | | One test episode, no baseline, winner file not saved. |
| C5 | code | Critical | todo | | `test.py` does not count trades or spread cost. |
| C6 | config | Critical | todo | | Hyperparameters are spread over scripts; no config file. |
| M1 | code | Major | todo | | Order book has no effect on agent fills. |
| M3 | code | Major | todo | | Entropic utility applied to each step. |
| M4 | config | Major | todo | | γ = 0.99 is too low for a 1,000-tick horizon. |
| B1 | code | Major | todo | | No seeds in workers; `srand(time(0))` in C++. |
| B2 | code | Major | todo | | Critic loss is much larger than actor loss. |
| B3 | code | Minor | todo | | One episode for each PPO update. |
| B4 | code | Minor | todo | | Terminal bootstrap value calculated but not used. |
| B5 | code | Minor | todo | | `main.cpp` is not used and has defects. |
| B6 | code | Minor | todo | | `rand() % 1000` modulo bias. |
| B7 | code | Minor | todo | | Network runs on one state at a time. |
| B8 | code | Minor | todo | | `setup.py` path, `train.py` cannot run. |
| C6 | comment | Critical | todo | | Comments say "3 Features"; the state has 4. |
| M5 | thesis | Major | deferred | | deferred: thesis. Hawkes is Markov; PPO has no guarantee. |

## Deferred: thesis

The thesis PDF is out of scope for this pass. These thesis parts are deferred.

| ID | Thesis location | Action |
|---|---|---|
| C1 | Section 4.4 | deferred: thesis. Change "PnL stabilizes at ~$19,000". |
| C2 | Fig. 2.1, Section 2.3 | deferred: thesis. Make Fig. 2.1 from the C++ engine. |
| C3 | Section 1.6.4, 4.3, Chapter 5 | deferred: thesis. "POMDP with delayed observations" to "MDP with delayed actions". |
| C4 | Chapter 4 | deferred: thesis. Report multi-seed metrics and baselines. |
| C5 | Section 4.4, 4.4.1 | deferred: thesis. Remove "damped harmonic oscillator" claim. |
| C6 | Table 3.1, Section 3.4.1, 4.2.1, 4.3, 4.4 | deferred: thesis. Make each row agree with the code. |
| M2 | Title, Chapter 1 | deferred: thesis. Inventory control, or add an option. |
| M4 | Section on γ | deferred: thesis. Remove "time-value of money" at 1 ms. |
| M5 | Section 1.6.3, 1.7.3, 2.3 | deferred: thesis. Correct the Markov and PPO statements. |
| M6 | Title, Chapter 3 | deferred: thesis. Say the cores are independent seeds. |

## Blocked: local run

Items that need a checkpoint or a training run are listed here.

## Math decisions

Decisions from the math-fixer agent are listed here.
