# Corrections Log

Source: `docs/AUDIT_REPORT_v2.md` (audit of commit `13391e9`).
Branch: `audit-corrections`, made from `master`.

Status values: todo, in progress, fixed, blocked, deferred.

## Findings

| ID | area | severity | status | files changed | note |
|---|---|---|---|---|---|
| B8 | README | Minor | fixed | README.md | Real clone URL, `requirements.txt`, no `.pyd`, v1 marked historical, tests step. |
| C1 | README | Critical | fixed | README.md | User decision: show both. ~$19,000 labelled as MTM incl. initial inventory; ≈ −$6,400 labelled as audit estimate. |
| C3 | README | Critical | fixed | README.md | User decision: audit wording. "MDP with delayed actions"; no ablation; hidden quantities named. |
| C4 | README | Critical | fixed | README.md | One episode, no baseline stated; new metrics marked "pending a local run". |
| C5 | README | Critical | fixed | README.md | User decision: audit wording. "Decaying limit cycle in one episode". |
| C6 | README | Critical | fixed | README.md | Three-column table: config.py default and thesis-code value. |
| M1 | README | Major | fixed | README.md | Simulator rules and Known limits state no impact, no liquidity limit. |
| M2 | README | Major | fixed | README.md | "Model contains no option" in section 1 and Known limits. Title kept (thesis title). |
| M6 | README | Major | fixed | README.md | Cores are independent runs; "asynchronous" refers to execution. |
| C1 | code | Critical | fixed | python/environment.py, python/test.py, python/train_multicore.py, tests/test_environment.py | Portfolio value includes inventory; penalty in reward only; shortfall reported. |
| C2 | code | Critical | fixed | src/exchange.cpp, python/generate_plot.py, tests/test_exchange.py | User decision: corrected default alpha_jump = 0.012; probability limited to 1; rate test. |
| C3 | code | Critical | blocked | python/agent.py, python/environment.py, python/config.py, python/train_multicore.py | Ablation code added (`--arch mlp`, `--no-in-flight`). blocked: local run (20 training runs). |
| C4 | code | Critical | blocked | python/evaluate.py, python/train_multicore.py, python/test.py | 1,000 test seeds, rule and TWAP baselines, validation winner. Agent metrics blocked: local run. |
| C5 | code | Critical | fixed | python/environment.py, python/test.py | Fills, spread cost and penalty reported. |
| C6 | config | Critical | fixed | python/config.py, python/agent.py, python/environment.py, python/train_multicore.py | User decision: README correct. Entropy 0.01, T = 64, MLP encoder, uniform I0. LayerNorm (thesis only) not added. |
| M1 | code | Major | deferred | src/exchange.cpp (comments only) | Book-based fills are a model change. Needs a user decision and a retrain. Comments document the limit. |
| M3 | code | Major | fixed | python/train_multicore.py, python/config.py | User decision: corrected default. Linear reward; `entropic_step` still selectable. |
| M4 | config | Major | fixed | python/config.py | User decision: γ = 0.999. math-fixer recommends 1.0 (open option, see Math decisions). |
| B1 | code | Major | fixed | src/exchange.cpp, python/environment.py, python/train_multicore.py | mt19937 with seed; torch, numpy, random seeds per core; seed per episode. |
| B2 | code | Major | fixed | python/train_multicore.py, python/config.py | Reward scale by running std of return; critic/actor loss ratio in log. |
| B3 | code | Minor | fixed | python/train_multicore.py, python/config.py | 16 environments per update; 4 minibatches of full sequences. |
| B4 | code | Minor | fixed | python/train_multicore.py | Terminal bootstrap call removed; `next_val = 0.0`. |
| B5 | code | Minor | fixed | legacy/main.cpp, legacy/README.md | Moved to `legacy/`. |
| B6 | code | Minor | fixed | src/exchange.cpp | `std::uniform_*_distribution` replaces `rand() %`. |
| B7 | code | Minor | fixed | python/agent.py, python/train_multicore.py, python/evaluate.py, src/exchange.cpp, python/environment.py | One forward pass for all environments of a core. `MarketBatch` steps all simulators in one C++ call: rollout 3.73 s to 0.89 s for 16 × 1,000 ticks. Results are identical (`tests/test_vector_env.py`). |
| B8 | code | Minor | fixed | python/setup.py, legacy/train_v1.py, requirements.txt, .gitignore | Build path, pinned versions, v1 script moved. |
| C6 | comment | Critical | fixed | python/agent.py, python/train_multicore.py, src/exchange.cpp | "3 Features" corrected to 4; market-order comment added. |
| M5 | thesis | Major | deferred | | deferred: thesis. Hawkes is Markov; PPO has no guarantee. README has no such statement. |

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

These items need the local checkpoints or new training runs. Run all commands from `python/` after the Setup steps in README.md.

| ID | Command | Inputs | Expected output |
|---|---|---|---|
| C1, C2, C6, M3, M4, B2 | `python train_multicore.py --run-name main` | none (new weights; old checkpoints do not load) | `runs/main/config.json`, `runs/main/weights_core_0..9.pth`, `runs/main/winner.json`, `deep_hedging_weights.pth`, `deep_hedging_config.json`. Log lines show `critic/actor loss` near 1, not much greater. |
| C4 | `python evaluate.py --policies agent rule twap --out results/main` | `deep_hedging_weights.pth`, `deep_hedging_config.json` | `results/main/eval_summary.json`: mean, std, CVaR 5% shortfall, RMS inventory, % ticks outside band, fills for each policy. |
| C3 | see loop below | none | 20 folders `runs/<arch>_<feat>_s<k>` and `results/<arch>_<feat>_s<k>/eval_summary.json`. |
| C5 | `python test.py` | `deep_hedging_weights.pth` | Shortfall, legacy MTM PnL, orders, fills, spread cost, penalty; new `thesis_v2_asynchronous_evaluation.png`. |
| C1 (legacy figure) | not possible as is | thesis checkpoint | The thesis code had no seed control (`srand(time(0))`), so the Fig. 4.3 episode cannot be run again. The −$6,400 value stays an audit estimate. |

C3 ablation loop (bash):

```bash
for arch in lstm mlp; do for feat in inflight noinflight; do for k in 0 1 2 3 4; do
  name=${arch}_${feat}_s${k}; flag=""; [ "$feat" = noinflight ] && flag="--no-in-flight"
  python train_multicore.py --run-name $name --arch $arch $flag --base-seed $((k*100))
  python evaluate.py --policies agent rule twap --out results/$name
done; done; done
```

`--base-seed $((k*100))` keeps the core seeds of different runs apart (core j uses base seed + j).

## Math decisions

Source: math-fixer agent (opus). It read the code and the audit only. It did not edit files or run code.

| ID | Decision | Reason | Changes a result |
|---|---|---|---|
| C1 | Previous value = cash + holdings × price at reset; penalty in reward only; report final MTM − initial MTM. Reporting fix has no flag. | The initial inventory is given, so its value is not a reward. The penalty is shaping, not cash. | Yes: all PnL values, Fig. 4.3 panel 3, winner choice. |
| C2 | alpha_jump = 0.012; λ = min(λ, 1). | The code decays first, then adds α. So n = αd/(1−d): 7.35 for 0.15, 0.588 for 0.012. Stable rate μ/(1−n) = 0.0485. The audit's 7.5 and 0.6 use α/(1−d). | Yes: all dynamics. The cap alone changes nothing. |
| M3 | Linear reward r/1000; keep `entropic_step` as an option. | A sum of step utilities is not the entropic risk of terminal PnL. At normal step sizes the function is almost linear. | Yes, small. |
| M4 | Recommended γ = 1.0; γ = 0.999 used (user approved 0.999). | Finite horizon and time in the state, so the undiscounted sum is the true objective. 0.999 still weights the last tick by 0.37. | Yes. Open option for the user. |
| B2 | Scale rewards by a running std of the discounted return (no mean shift). | A positive scale keeps the optimal policy. Advantages are already normalized. | Yes: training dynamics. |
| B4 | Remove the call; `next_val = 0`. | The value was multiplied by zero. The removed call used the torch RNG, so runs are not bit-identical, but the distribution is the same. | No. |

## Pilot runs

| Pilot | Commit | Setting | Validation result (seeds 5000-5099) |
|---|---|---|---|
| 1 | `1e1e406` | 1 core, 2,000 episodes, ppo_epochs 4 | Agent: shortfall −$5,291, 873 fills, 84% ticks outside band. Rule: −$107, 22 fills, 7%. Over-trading limit cycle (C5). |
| 2 | `cf1b9dd` | 1 core, 8,000 episodes, ppo_epochs 10 (user decision) | Agent: shortfall −$3,603, penalty 5,062, 596 fills, 59% outside band; J = shortfall − penalty = −8,665. Rule: J = −4,353. Critic explained variance 0.93-0.97. Entropy 0.94-1.03 (max 1.099). Hold fraction 0.29 to 0.40-0.50. |

| 3A | `d28ad0c` | as pilot 2, plus cost shaping (user decision) | Agent: shortfall −$219, penalty 5,036, J = −5,255, 38.5 fills, 68.5% outside band. ΔJ vs rule −902 (95% CI −1,200 to −644). FAIL (fills pass; J and band fail). Entropy still falling at the end (0.51). |
| 3B | `d28ad0c` | as pilot 3A, plus entropy_coef 0.001 (user decision) | Agent: shortfall −$253, penalty 4,667, J = −4,920, 34.0 fills, 79.0% outside band. ΔJ vs rule −567 (95% CI −727 to −426). FAIL (fills pass; J and band fail). Entropy 0.04-0.10 from episode 6,400. |
| 4A | `d28ad0c` | pilot 3A weights, 16,000 more episodes (24,000 in total), base seed 1000 (user decision). Adam state and reward normalizer restart. | Agent: shortfall −$162, penalty 4,281, J = −4,442, 33.5 fills, 13.0% outside band, RMS inventory 74.0. ΔJ vs rule −90 (95% CI −146 to −33). **PASS** (all three criteria). Not better than the rule (CI below 0). Explained variance 0.98-0.99. |
| 5 | `8fa1eb2` | as pilot 3A, plus penalty shaping (user decision), new weights, 8,000 episodes | Agent: shortfall −$185, penalty 4,525, J = −4,711, 30.7 fills, 61.4% outside band. ΔJ vs rule −358 (95% CI −434 to −285). FAIL (fills pass; J and band fail). Critic explained variance fell from 0.92 to 0.68. |
| 5L | `8fa1eb2` | pilot 5 weights, 16,000 more episodes (24,000 in total), base seed 1000 (user decision). Same method as 4A. | Agent: shortfall −$164, penalty 4,609, J = −4,772, 28.6 fills, 50.8% outside band, RMS inventory 81.8. ΔJ vs rule −420 (95% CI −594 to −271). FAIL (fills pass; J and band fail). Worse than pilot 5 at 8,000 episodes. Entropy 0.02-0.09 (4A: 0.10-0.20). Explained variance 0.66-0.88 (4A: 0.97-0.99). |

Pilot 1 had 2,000 gradient steps. The thesis code had 32,000. ppo_epochs is now 10 (user decision).

total_episodes is now 24,000 for each core (user decision). The only passing pilot (4A) used 24,000 episodes. At 8,000 episodes, no setting passed.

Pilot 5L compares the two shaping options at 24,000 episodes. Cost shaping only (4A) passes. Cost and penalty shaping (5L) fails, and it did not improve after episode 8,000. With penalty shaping, the policy entropy fell below 0.1 and the critic explained variance stayed below 0.9. The policy then had almost no exploration to correct its band errors.

### Pilot 2 diagnosis (math-fixer, opus)

- Main cause: the `holdings × Δprice` term adds price noise to each step reward. One order's cost ($6) is small compared with this noise.
- The cost reaches the reward at the fill, 50 ticks after the decision. Through GAE (λ = 0.95, γ = 0.999) it has weight 0.073.
- Entropy stays near the maximum, so the greedy policy chooses from almost equal logits.
- Rejected: GAE λ = 1.0. It adds up to 1,000 ticks of price noise to each advantage.
- Proposed (pilot 3): potential-based shaping with Φ(s) = −pending_cost / reward_scale. pending_cost = $6 × (orders placed) − spread and fees paid. Φ(s_0) = Φ(s_T) = 0, so the optimal policy and the objective do not change (Ng et al., 1999).
- Second option if entropy stays above 0.9: entropy_coef 0.001.
- Pass criterion: paired bootstrap 95% CI lower bound of J(agent) − J(rule) above −218, mean fills ≤ 66, and ticks outside the band ≤ rule + 10 points.

### Pilot 3 diagnosis: penalty shaping (math-fixer, opus)

- Cause: the cost shaping charges an order's $6 at the decision tick. Its penalty benefit reaches the advantage 49 ticks later, with weight 0.074.
- With this asymmetry, an order is worth its cost only when the excess inventory is above about 210 shares. This explains why the agent parks 40-80 shares outside the band.
- Proposed: Φ = −(pending_cost + projected_penalty) / reward_scale. projected_penalty is the discounted penalty of the next 49 ticks. Orders already sent fix these ticks.
- Φ uses holdings and the agent's own past actions (observed history). The identity Σγ^t F_t = γ^T Φ(s_T) − Φ(s_0) holds on every path, so the optimal policy does not change.
- Φ(s_0) ≠ 0 because I0 is outside the band. It depends only on I0, so it is a constant for each episode. Start with phi_prev = Φ(s_0).
- Telescoped form: each tick the agent pays −γ^49 × penalty(holdings + in_flight). This is the quantity the rule baseline uses.
- Fill delay verified in the cloud: an order placed at step 3 fills in step 52 (k + 49).
- Status: implemented (user decision). tests/test_penalty_shaping.py checks the telescoping, the closed form and the policy invariance.

## Process notes

- `main` is named `master` in this repository. The branch was made from `origin/master`.
- New agent files in `.claude/agents/` do not load in a running session. The agents ran as general-purpose agents with the same instructions and the model from the frontmatter.
- The main session applied the code fixes directly, not through code-fixer, to keep one commit for each finding group.
- `download.pytorch.org` is blocked by the cloud network policy. torch came from PyPI.
- Smoke tests (each under 60 s): build, 9 pytest tests, baselines on 20 seeds, 1 core × 48 episodes training, `test.py`, `generate_plot.py`. Smoke numbers are not results.
