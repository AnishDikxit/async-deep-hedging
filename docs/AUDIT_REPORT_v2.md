# Audit Report: async-deep-hedging (v2, PPO-LSTM)

| Item | Data |
|---|---|
| Repository | `github.com/AnishDikxit/async-deep-hedging` |
| Commit | `13391e9` (2026-05-27, "Modified README") |
| Thesis | *Asynchronous Actor-Critic Architectures for Delta-Neutral Hedging*, IIT (BHU), 2026, 56 pages |
| Audit date | 2026-10-10 |
| Writing standard | ASD-STE100 Simplified Technical English (STE) |

**Note on the writing standard.** This report uses STE rules for all text. Code, file names, equations and quotations from the thesis are technical names. They stay in their initial form.

---

## 1. Scope

1. This audit covers the v2 system only. The v2 system is the asynchronous environment with the PPO-LSTM agent.
2. These files are in scope:

   | File | Function |
   |---|---|
   | `src/exchange.cpp` | C++ market simulator with the 50-tick latency queue (Pybind11 module `cpp_exchange`) |
   | `python/environment.py` | Python environment. It sends orders and calculates the reward. |
   | `python/agent.py` | `ActorCriticLSTM` network and `PPOAgentLSTM` agent |
   | `python/train_multicore.py` | Main training script (PPO with GAE, 10 independent processes) |
   | `python/test.py` | Evaluation script and result figure |
   | `python/setup.py` | Build script for the C++ module |
   | `python/generate_plot.py` | GBM and Hawkes comparison figure (thesis Fig. 2.1) |
   | `README.md` | Project documentation |
   | Thesis Chapters 1, 2 and 4, Table 3.1, Appendices A and B | Claims about the v2 system |

3. The v1 system (synchronous feed-forward agent) is out of scope. Its results were not kept, and its code was changed to make v2.
4. The audit refers to v1 files only for repository cleanup (finding B8).

### 1.1 Method

1. We read all source files and all 56 pages of the thesis.
2. We compared each claim in the thesis and README with the code.
3. We wrote a Python copy of the Hawkes logic in `exchange.cpp`. We ran it for 2,000 episodes (Appendix A).
4. We calculated costs and rewards from the code. We compared these values with the v2 result figure (`assets/thesis_v2_asynchronous_evaluation.png`, thesis Fig. 4.3).
5. We did not compile the C++ module. We did not run the training script.

### 1.2 Severity levels

| Level | Definition |
|---|---|
| **Critical** | The problem changes a main conclusion of the thesis. |
| **Major** | The problem makes a model, an equation or a statement incorrect. The main conclusion stays possible. |
| **Minor** | The problem decreases quality, speed or reproducibility. |

---

## 2. Summary of findings

| ID | Severity | Title | Effect |
|---|---|---|---|
| C1 | Critical | The environment does not include the initial inventory in the portfolio value | Reported PnL is not true PnL. The first reward is very large. |
| C2 | Critical | The Hawkes process is unstable (supercritical) | The market becomes a simple random walk. It has no volatility clusters. |
| C3 | Critical | Two changes were tested together, with no ablation | The data does not show that the LSTM is necessary. |
| C4 | Critical | The evaluation uses one episode and no baseline | The data cannot support a general claim. |
| C5 | Critical | The figure does not agree with its description | The agent trades on most ticks. Inventory continues to oscillate. |
| C6 | Critical | The thesis does not agree with the code | The thesis cannot be reproduced from its text. |
| M1 | Major | The order book has no effect on the agent | There is no market impact and no liquidity limit. |
| M2 | Major | The model contains no option | "Delta-neutral" has no exact meaning here. |
| M3 | Major | The entropic utility is applied to each step | Risk aversion is almost zero. |
| M4 | Major | The discount factor is too low for the horizon | The agent ignores most of the episode. |
| M5 | Major | Some theory statements are incorrect | Hawkes is Markov. PPO has no guarantee. |
| M6 | Major | "Asynchronous" and "distributed" are used in a misleading way | The cores do not share data. |
| B1 | Major | All workers can start with the same random seeds | The 10 cores are not independent. The runs cannot be reproduced. |
| B2 | Major | The critic loss is much larger than the actor loss | The actor receives little gradient. |
| B3 | Minor | Each update uses only one episode | The gradient estimates have high variance. |
| B4 | Minor | A terminal bootstrap value is calculated but not used | The calculation is unnecessary. |
| B5 | Minor | `main.cpp` is not used and has defects | Old code causes confusion. |
| B6 | Minor | `rand() % 1000` has a modulo bias | The event probabilities are slightly incorrect. |
| B7 | Minor | The network runs on one state at a time | Training is slow. |
| B8 | Minor | Build and repository problems | A new user cannot build or run the project from the README. |

---

## 3. Correct items (keep these)

1. The GAE recursion in `train_multicore.py:64-76` is correct. It does not bootstrap at the terminal step. This is correct because the state includes the remaining time.
2. The agent sets the LSTM state to zero at the start of each episode (`train_multicore.py:40`).
3. The update calculates all hidden states again from zero with the current weights (`agent.py:67`). This prevents errors from old hidden states.
4. The environment calculates `in_flight_exposure` correctly when it sends an order and when it receives a fill.
5. The training script normalizes the advantages and clips the gradient norm.
6. The C++ fill price is the price 50 ticks after the decision. This correctly models execution latency.

---

## 4. Critical findings

### C1. The environment does not include the initial inventory in the portfolio value

**Location:** `environment.py:20-23`, `environment.py:69`, `test.py:51`

**Description**
1. `reset()` gives the agent a random inventory of -500, -250, +250 or +500 shares.
2. `reset()` sets `previous_portfolio_value` to the cash value only. It does not add `holdings × price`.
3. `test.py` reports `cash + holdings × price - starting_cash` as PnL. This value includes the market value of the initial shares.
4. `step()` subtracts the tolerance-band penalty from `cash`. Thus the reported PnL also includes the penalty. The penalty is a training signal, not a cash flow.

**Effect**
1. At t = 0, the reported PnL is ±$25,000 or ±$50,000. The agent did nothing to cause this value.
2. The first reward of each episode is equal to `holdings × price`. After the entropic transform, the first reward has these values:

   | Initial inventory | Raw first reward | Utility (λ = 0.05) |
   |---|---|---|
   | +500 | +$50,000 | +18.4 |
   | +250 | +$25,000 | +14.3 |
   | -250 | -$25,000 | -49.8 |
   | -500 | -$50,000 | -223.6 |

   A typical step reward has a utility between 0.001 and 0.2. The action cannot change the first reward. The first reward controls the critic loss (refer to B2).

3. The thesis states "PnL stabilizes at ~$19,000". The true loss is approximately -$6,400 (refer to Evidence).

**Evidence**
1. In Fig. 4.3, the initial inventory is +250 shares. The PnL curve starts at 25,000 (250 × $100).
2. The curve ends at approximately 18,600. The true trading loss is approximately 25,000 - 18,600 = $6,400.
3. During the first 50 ticks, no order can fill. The penalty for each tick is 0.001 × (250 - 30)² = $48.40. For 50 ticks, the penalty is $2,420. The figure shows a decrease from 25,000 to approximately 22,800 at tick 50. The penalty causes almost all of this decrease.

**Corrective action**
1. In `reset()`, after `self.engine.step()`, set `self.previous_portfolio_value = self.cash + self.holdings * self.current_price`.
2. In `reset()`, keep this value as `self.initial_mtm`.
3. In `step()`, do not subtract the penalty from `self.cash`.
4. In `step()`, calculate `reward = (portfolio_value - previous_portfolio_value) - inventory_penalty`.
5. Report the implementation shortfall: `final_mtm - initial_mtm`. This value contains only market gains, losses and costs.
6. Show the penalty as a separate metric.

```python
# environment.py — corrected parts
def reset(self, seed=None):
    ...
    self.current_price, _ = self.engine.step()
    self.previous_portfolio_value = self.cash + self.holdings * self.current_price
    self.initial_mtm = self.previous_portfolio_value
    return self._get_state()

def step(self, action):
    ...
    excess = max(0, abs(self.holdings) - 30)
    inventory_penalty = 0.001 * excess ** 2          # reward only, not cash
    portfolio_value = self.cash + self.holdings * self.current_price
    reward = (portfolio_value - self.previous_portfolio_value) - inventory_penalty
    self.previous_portfolio_value = portfolio_value
    ...
```

---

### C2. The Hawkes process is unstable (supercritical)

**Location:** `exchange.cpp:47-49`, `exchange.cpp:112-118`, `generate_plot.py`

**Description**
1. The engine uses `base_mu = 0.02`, `alpha_jump = 0.15` and `decay_rate = 0.98`.
2. Each event adds 0.15 to the intensity. This addition decreases by 2% on each tick.
3. The mean number of new events that one event causes is the branching ratio: 0.15 / (1 - 0.98) = **7.5**.
4. A Hawkes process is stable only when the branching ratio is less than 1.
5. The engine uses the intensity as a probability: `(rand() % 1000) / 1000.0 < λ`. When λ is more than 1, an event occurs on each tick.
6. When events occur on each tick, the excitation goes to its fixed point of 0.15 / 0.02 = 7.5. Thus the market cannot become calm again.

**Effect**
1. After the lock-in, the price changes on each tick by a random integer from -2 to +2. Each change is independent.
2. Thus the market is a simple random walk (a discrete Brownian motion). The thesis states that it replaces this type of model.
3. The market has no volatility clusters, no calm periods and no flash crashes.
4. Thesis Fig. 2.1 does not show the engine. `generate_plot.py` uses a different model (μ = 0.01, α = 0.08, β = 0.15, Gaussian jumps with σ = 1.8). That model has a branching ratio of approximately 0.58. It is stable.

**Evidence**
1. We copied the Hawkes logic to Python. We ran 2,000 episodes of 1,000 ticks (Appendix A).

   | Result | Current engine (α = 0.15) | Corrected engine (α = 0.012) |
   |---|---|---|
   | Branching ratio | 7.5 | 0.6 |
   | Mean events in 1,000 ticks | 941 | 46 |
   | Episodes with an event on each tick | 100% | 0% |
   | Median lock-in tick | 49 | none |

2. In Fig. 4.3, the price is flat for approximately 50 ticks. After that, it changes on almost each tick with a constant size. This agrees with the simulation.

**Corrective action**
1. Set `alpha_jump` so that `alpha_jump / (1 - decay_rate)` is less than 1. For example, set `alpha_jump = 0.012`.
2. Limit the probability to 1.0 explicitly in the code.
3. Make Fig. 2.1 from the output of the C++ engine, not from `generate_plot.py`.
4. Add a test that measures the mean event rate. The stable value is `base_mu / (1 - branching_ratio)` for each tick.

---

### C3. Two changes were tested together, with no ablation

**Location:** Thesis Section 4.3, Chapter 5; `environment.py:80-88`; `agent.py:13`

**Description**
1. The v2 system adds two changes at the same time:
   - the state feature `in_flight_exposure` (E_t)
   - the LSTM layer
2. The thesis gives the improvement to the LSTM ("necessitating the use of recurrent neural architectures"). No experiment tests each change alone.
3. Theory shows that E_t alone can be sufficient. An MDP with delayed actions becomes Markov again when the state includes the pending actions (Katsikopoulos and Engelbrecht, 2003). The sum `holdings + in_flight_exposure` is the inventory after all pending fills. A feed-forward network can use this sum to stop trading at the correct time. It does not need memory.
4. The thesis states that the observation is delayed ("stale price", Section 1.6.4). This is not correct. The agent sees the current price and the current inventory (`environment.py:47`). Only the execution is delayed. Thus the model is an MDP with delayed actions. It is not a POMDP with delayed observations.
5. Two quantities are hidden from the agent:
   - the fill time of each pending order (E_t gives only the total)
   - the Hawkes excitation level
   These hidden quantities are the correct reason to use an LSTM. They are also testable.

**Effect**
The data does not show that the LSTM is necessary. The main claim of Chapter 4 is not supported.

**Corrective action**
1. Make a feed-forward version of the agent with the same hidden size.
2. Train four configurations: {MLP, LSTM} × {with E_t, without E_t}.
3. Train each configuration with a minimum of 5 seeds.
4. Evaluate all configurations on the same test seeds (refer to C4).
5. Change the thesis text from "POMDP with delayed observations" to "MDP with delayed actions". Name the hidden quantities in item 5 above.

---

### C4. The evaluation uses one episode and no baseline

**Location:** `test.py`, `train_multicore.py:115-135`

**Description**
1. `test.py` runs one episode of 1,000 ticks. All of Chapter 4 uses this one episode.
2. The evaluation has no baseline policy. A simple rule can do this task:
   ```text
   if holdings + in_flight > 30: sell
   elif holdings + in_flight < -30: buy
   else: hold
   ```
3. The script selects the "winner" core from the mean training PnL of the last 100 episodes. This PnL includes the random initial value from C1.
   - The standard deviation of the initial value for each episode is √((25,000² + 50,000²) / 2) ≈ $39,500.
   - Thus the standard deviation of a 100-episode mean is approximately $3,950.
   - Most differences between cores are random. The selection of the maximum of 10 values also adds a positive bias.
4. `train_multicore.py` saves `weights_core_N.pth`. `test.py` loads `deep_hedging_weights.pth`. No script copies the winner file. The repository does not record which weights made Fig. 4.3.

**Effect**
1. One episode cannot show a general property of the policy.
2. Without a baseline, the reader cannot know whether the agent is good.

**Corrective action**
1. Make a fixed list of 1,000 test seeds. Use the same seeds for all policies.
2. Run each policy on all test seeds.
3. Report these metrics for each policy:
   - mean and standard deviation of the implementation shortfall
   - CVaR at 5% of the implementation shortfall
   - RMS inventory
   - percentage of ticks outside the ±30 band
   - number of trades
4. Add two baselines: the rule in item 2 above, and TWAP liquidation.
5. Select the best core with a separate validation seed set, not with training PnL.
6. Save the selected weights to the file that `test.py` loads. Record the core ID and seeds in a log file.

---

### C5. The figure does not agree with its description

**Location:** Thesis Section 4.4 and 4.4.1, Fig. 4.3, README "v2.0 Success"

**Description**
1. The thesis states that the agent "pauses execution while it waits for the 50ms queue to clear".
2. The trade markers in Fig. 4.3 show orders on most ticks.
3. The inventory curve is a triangle wave. Its slopes are ±10 shares per tick. This is the maximum trade rate.
4. The inventory overshoots to ±100 to ±300 shares. The overshoots continue to the end of the episode.
5. The thesis calls this a "damped harmonic oscillator". The audit found no fit of a decay rate or damping ratio. The data is one episode.

**Effect**
1. The figure shows the same latency limit cycle as the failed agent, with a smaller amplitude.
2. The agent has E_t in its state. A policy that uses E_t correctly has almost no overshoot. Thus the agent did not learn to use E_t fully.
3. Each trade costs $6 ($0.50 × 10 shares + $1 fee). Our estimate from the figure is approximately 900 trades. These trades cost approximately $5,400. This is most of the $6,400 loss in C1. Thus the agent loses money mostly because it trades too much.

**Corrective action**
1. Count the trades and the spread cost in `test.py`. Report them.
2. Describe the result as "a decaying limit cycle in one episode", or measure the decay over many episodes.
3. Compare the result with the baseline rule from C4.

---

### C6. The thesis does not agree with the code

**Location:** Thesis Table 3.1, Sections 3.4.1, 4.3 and 4.4; README

| Thesis or README states | Code does |
|---|---|
| Truncated BPTT with T = 64 (Section 4.3, README) | Full BPTT over 1,000 steps. `evaluate()` runs the full sequence from a zero state (`train_multicore.py:89-92`). |
| The MLP encoder sends features to the LSTM (Section 4.3) | The 4-feature state goes directly into the LSTM (`agent.py:13`). There is no MLP encoder and no LayerNorm. |
| Learning rate 3 × 10⁻⁴ (Table 3.1) | 1 × 10⁻⁴ (`train_multicore.py:27`) |
| Entropy coefficient 0.01 (Table 3.1, README) | 0.05 (`train_multicore.py:101`) |
| Initial inventory I₀ ~ U(-500, 500) | `random.choice([-500, -250, 250, 500])` |
| Training changes the Hawkes μ and α by small amounts | μ and α are `const double` values |
| Test environments are "deterministically seeded" | `srand(time(0))`. There is no seed control. |
| Tests use higher α values | No code does this. |
| Reward is Eq. 3.3 with risk-aversion parameter λ | The penalty coefficient is 0.001 in the code. The code also applies an entropic utility with λ = 0.05. The thesis does not show this transform. |
| The agent sends "limit orders" (Section 4.2.1, 4.4) | The orders are market orders. The C++ fill ignores the `price` argument (`exchange.cpp:144`). |
| A "pending memory ledger" decouples execution | Correct |
| 50 ms latency | Correct (50 ticks, with 1 tick = 1 ms) |

**Corrective action**
1. Change the thesis or the code so that each row agrees.
2. Keep all hyperparameters in one configuration file. Write this file to the output folder for each run.

---

## 5. Major findings (model and mathematics)

### M1. The order book has no effect on the agent

**Location:** `exchange.cpp:51-71`, `exchange.cpp:136-151`

1. The engine adds random orders to `bids` and `asks`. These orders match only with other random orders.
2. The book does not change the price. The book does not change the agent fills.
3. The agent orders do not go into the book. The engine fills each agent order completely at `current_market_price ± $0.50`.
4. The `else add_order(ready_order)` branch never runs. Random orders never go into the latency queue.
5. Thus the model has no market impact, no liquidity limit and no variable spread. The slippage is a fixed half-spread plus the price change during 50 ticks. The mean of this price change is zero.

**Corrective action**
1. Fill agent orders against the book. Use partial fills when the book has insufficient volume.
2. Add a price impact that increases with order size.
3. If you do not use the book, remove it. Change the thesis text about liquidity and spreads.

### M2. The model contains no option

1. Chapter 1 explains options, Black-Scholes and Delta.
2. The environment contains no option. The agent moves a stock inventory to zero.
3. This task is optimal liquidation and inventory control (Almgren-Chriss; Cartea, Jaimungal and Penalva).

**Corrective action**
Do one of these two steps:
1. Change the thesis title and text to "inventory risk control under execution latency".
2. Add a short call option to the portfolio. Set the hedge target to -Δ_BS(S, t). Apply the penalty to the difference between the inventory and the target.

### M3. The entropic utility is applied to each step

**Location:** `train_multicore.py:18-22, 50`

1. The entropic risk measure is ρ(X) = (1/λ) log E[e^(-λX)]. X is the terminal PnL.
2. The code applies u(r) = (1 - e^(-λ r / 1000)) / λ to each step reward. The sum of these utilities is not the entropic risk of the terminal PnL.
3. After the division by 1000, a typical step reward gives λr/1000 between 10⁻⁵ and 10⁻². For these values, u(r) is almost equal to r/1000. Thus the risk aversion is almost zero. Only the first step (C1) is in the non-linear part.

**Corrective action**
1. Use a linear reward for each step.
2. Apply a risk measure to the terminal PnL. Use a critic-free method (Section 8.3) or a distributional critic.

### M4. The discount factor is too low for the horizon

1. γ = 0.99 for each 1 ms tick. The effective horizon is approximately 1 / (1 - γ) = 100 ticks.
2. 0.99¹⁰⁰⁰ ≈ 4 × 10⁻⁵. Thus the agent gives almost no value to the second half of the episode.
3. The thesis states that γ reflects "the time-value of money". At a 1 ms scale, this is not correct.

**Corrective action**
1. Set γ = 1.0 or γ = 0.999. The task has a finite horizon. The state includes the remaining time.

### M5. Some theory statements are incorrect

1. Sections 1.6.3 and 2.3 state that the Hawkes process is non-Markovian. With an exponential kernel, the intensity λ(t) is a Markov process.
2. Section 1.7.3 states that PPO "mathematically guarantees" stable learning and that the agent "cannot accidentally unlearn". PPO has no such guarantee. TRPO has an approximate bound for monotonic improvement. The PPO clip is a heuristic. It decreases the probability of large policy changes.

**Corrective action**
Change these statements in the thesis.

### M6. "Asynchronous" and "distributed" are used in a misleading way

1. Each process trains a separate agent. The processes do not share gradients, weights or data.
2. Thus the system is 10 independent training runs. It is not a distributed rollout system for one agent.
3. The title "Asynchronous Actor-Critic" is similar to A3C. A3C uses asynchronous workers that share parameters. This project does not do this. Here, the word "asynchronous" refers to the market execution.

**Corrective action**
1. Write in the thesis that the "asynchronous" property is in the market.
2. Write that the cores are independent seeds.

---

## 6. Code findings

### B1. All workers can start with the same random seeds

**Location:** `exchange.cpp:82`, `train_multicore.py:24-27`

1. The C++ constructor calls `srand(time(0))`. Workers that start in the same second get the same price paths.
2. PyTorch sets its CPU generator to a fixed constant in each new process. The code does not call `torch.manual_seed`. Thus all 10 workers start with the same network weights and the same random sample sequence.
3. Only the Python `random` module (initial inventory) is different in each worker.

**Corrective action**
1. In each worker, call `torch.manual_seed(base_seed + worker_id)`, `np.random.seed(...)` and `random.seed(...)`.
2. In C++, replace `rand()` with `std::mt19937`.
3. Add a `seed` argument to the `MarketSimulator` constructor and to `reset()`.
4. Write all seeds to the run log.

### B2. The critic loss is much larger than the actor loss

**Location:** `agent.py:7-29`, `train_multicore.py:85, 99-105`

1. The actor and the critic use the same LSTM.
2. The value targets are not normalized. They include the first-step value from C1 (as low as -223.6).
3. Thus `0.5 * critic_loss` can be many times larger than `actor_loss`.
4. `clip_grad_norm_(..., max_norm=0.5)` decreases the total gradient. Thus the actor receives only a small part of the gradient.

**Corrective action**
1. Do the corrective action for C1 first.
2. Normalize the value targets with a running mean and standard deviation.
3. Alternatively, use a separate LSTM for the critic.

### B3. Each update uses only one episode

1. Each PPO update uses one episode of 1,000 correlated steps.
2. The update uses this data as one full batch for 4 epochs.
3. Thus the gradient estimate has high variance.

**Corrective action**
Collect data from N parallel environments for T steps. Divide the data into sequence chunks. Use minibatches of chunks (Section 7.4).

### B4. A terminal bootstrap value is calculated but not used

**Location:** `train_multicore.py:62, 68-69`
The script calculates `next_state_value` and then multiplies it by zero. The result is correct. The calculation is unnecessary. Remove it.

### B5. `main.cpp` is not used and has defects

1. `main.cpp` is an old Redis version of the engine. No script uses it.
2. It does not examine whether `reply` is `NULL`.
3. It runs a loop with no delay. This loop uses 100% of one CPU core.

**Corrective action**
Remove `main.cpp`, or move it to a `legacy/` folder.

### B6. `rand() % 1000` has a modulo bias

On Windows, `RAND_MAX` is 32767. Thus `rand() % 1000` gives values 0 to 767 more frequently than values 768 to 999. The error is small. The corrective action for B1 (`std::mt19937` with `std::uniform_real_distribution`) also removes this error.

### B7. The network runs on one state at a time

1. `select_action` sends a batch of one state to the network on each tick.
2. Python overhead controls the speed, not the GIL.

**Corrective action**
Run 32 to 64 environments in each process. Send all their states to the network in one batch.

### B8. Build and repository problems

1. `setup.py` compiles `exchange.cpp` from the `python/` folder. The file is in `src/`.
2. The README refers to `requirements.txt`. The repository does not contain this file.
3. The README refers to `src/cpp_exchange.cp312-win_amd64.pyd`. The repository does not contain this file.
4. The README clone command uses `yourusername`.
5. `train.py` is an old v1 script. It imports `RLAgent`, which is not in `agent.py`. It cannot run. The README still tells the user to run it.
6. The README describes v1 results and shows v1 figures. The repository has no code or data for them.

**Corrective action**
1. Change `setup.py` to use `../src/exchange.cpp`.
2. Add `requirements.txt` with exact versions (`torch`, `numpy`, `matplotlib`, `pybind11`).
3. Remove the references to the `.pyd` file, or add the file.
4. Change `yourusername` to `AnishDikxit`.
5. Remove `train.py`, or move it to `legacy/`.
6. Remove the v1 sections from the README, or mark them as historical.

---

## 7. Theory guide: BPTT and PPO-LSTM

This section explains the two algorithms with the v2 code.

### 7.1 The LSTM in the agent

1. On each tick t, the LSTM receives the state x_t = [price, holdings, time, in_flight].
2. The LSTM also receives its memory from the previous tick: the hidden state h_(t-1) and the cell state c_(t-1).
3. The LSTM calculates a new memory (h_t, c_t).
4. The actor head and the critic head read only h_t.
5. The LSTM uses the same weights W on each tick.
6. Thus h_t is a learned summary of all previous states. It is the "belief state" of the agent.

```
x_0 → [LSTM] → h_0 → π(·|h_0), V(h_0)
         ↓ (h_0, c_0)
x_1 → [LSTM] → h_1 → π(·|h_1), V(h_1)
         ↓
        ...
```

### 7.2 Backpropagation Through Time (BPTT)

1. The loss is a sum of terms from each tick: L = Σ_t ℓ_t(h_t).
2. h_t depends on W directly. h_t also depends on W through h_(t-1), h_(t-2) and all previous states.
3. The chain rule gives this gradient:

   ```
   ∂L/∂W = Σ_t Σ_(k≤t)  ∂ℓ_t/∂h_t · (∂h_t/∂h_(t-1) ··· ∂h_(k+1)/∂h_k) · ∂h_k/∂W
   ```

4. "Through time" has this meaning: the loop becomes one long feed-forward graph with 1,000 copies of the cell. All copies share the same weights. Standard backpropagation then runs on this graph.
5. PyTorch does this automatically. In the v2 code:

   ```python
   # agent.py:67 — one call on the full [1, 1000, 4] sequence makes a graph with 1000 steps
   action_probs, state_values, _ = self.policy_network(states, initial_hidden)
   # train_multicore.py:104 — this call does full BPTT through 1000 steps
   loss.backward()
   ```

**Why the LSTM helps BPTT**
1. In a simple RNN, the product of the Jacobians ∂h_t/∂h_(t-1) usually goes to zero or becomes very large over many steps.
2. The LSTM changes its cell state by addition: c_t = f_t ⊙ c_(t-1) + i_t ⊙ c̃_t.
3. Thus ∂c_t/∂c_(t-1) = diag(f_t).
4. When the forget gate f_t is near 1, the gradient goes back through many steps with almost no change.

**Truncated BPTT**
1. Truncated BPTT divides the sequence into chunks of K steps.
2. The forward pass sends the hidden values from one chunk to the next chunk.
3. The code stops the gradient at each chunk boundary with `.detach()`.
4. Thus the memory can contain data from the full episode. But the gradient teaches the network to keep data for a maximum of K steps.
5. The latency is 50 ticks. Thus K = 64 is a good value.
6. The thesis states K = 64, but the code does not use truncation (refer to C6). This is an example implementation:

   ```python
   def evaluate_tbptt(self, states, action_idx, h, K=64):
       logps, vals, ents = [], [], []
       for s in range(0, states.shape[1], K):
           probs, v, h = self.policy_network(states[:, s:s+K], h)
           h = (h[0].detach(), h[1].detach())          # stop the gradient here
           d = Categorical(probs.squeeze(0))
           logps.append(d.log_prob(action_idx[s:s+K]))
           ents.append(d.entropy())
           vals.append(v.squeeze())
       return torch.cat(logps), torch.cat(vals), torch.cat(ents)
   ```

**Two different credit-assignment problems**
The thesis mixes these two problems (Section 3.3.2). Learn to keep them separate.

| Question | Method that answers it |
|---|---|
| "The sell order at t = 10 filled at t = 60. Was the order good?" This gives credit from rewards to actions. | **Returns and GAE.** The advantage Â_t adds the discounted future rewards. This is arithmetic on stored values. It does not use BPTT. |
| "What must the LSTM keep in memory at t = 10 to act correctly at t = 40?" This gives credit from decisions to memory. | **BPTT.** The gradient goes back through h and changes what the LSTM keeps. |

### 7.3 PPO step by step (`train_multicore.py`)

1. **Collect data with the current policy π_old** (lines 45-56).
   - Keep s_t, a_t, log π_old(a_t|s_t), V_old(s_t) and r_t.
   - `torch.no_grad()` is correct here. These values are fixed data.
2. **Calculate the advantages with GAE** (lines 64-76):

   ```
   δ_t = r_t + γ V(s_(t+1)) - V(s_t)        # error of a 1-step prediction
   Â_t = δ_t + γλ Â_(t+1)                   # weighted sum of future errors
   ```

   - λ = 0 gives the 1-step error. It has low variance and a bias.
   - λ = 1 gives the Monte Carlo return minus V. It has no bias and high variance.
   - λ = 0.95 is between these two.
   - The value target is `returns = Â + V_old` (line 85).
3. **Calculate the probability of the same actions with the current weights θ** (line 92).
   - The ratio is r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t) = exp(logp - old_logp).
4. **Calculate the clipped objective** (lines 94-98).
   - The objective is the mean of min(r·Â, clip(r, 1-ε, 1+ε)·Â).
   - When Â > 0, the update increases the probability of the action. When r is more than 1.2, the gradient becomes zero.
   - When Â < 0, the update decreases the probability. When r is less than 0.8, the gradient becomes zero.
   - The `min` gives a pessimistic value. A large policy change does not receive more reward.
   - Example: Â = +2 and r = 1.5. The two terms are 3.0 and 1.2 × 2 = 2.4. The minimum is 2.4. This value is constant, so the gradient is 0.
5. **Calculate the total loss:** actor loss + c_v × (V - returns)² - c_e × entropy. The entropy term keeps exploration.
6. **Do steps 3 to 5 again for K epochs** on the same data. The clip makes this reuse of data safe.

### 7.4 Changes that the LSTM causes in PPO

1. **The hidden state is part of the input.** The policy is π(a_t | x_t, h_(t-1)). When θ changes during the PPO epochs, the new h_t is different from the h_t of the data collection. There are two solutions:
   - **Solution A (the v2 code uses this):** calculate all h_t again from h_0 = 0 with the current θ. This is exact. But you cannot mix timesteps in random order.
   - **Solution B:** keep h at the start of each chunk of K steps during data collection. During the update, run each chunk again from its stored h. You can add a short "burn-in" before each chunk. This solution allows minibatches of chunks from many environments.
2. **Do not put single timesteps in random order.** A minibatch must contain full sequences or full chunks.
3. **Set h to zero at the start of each episode.** The v2 code does this. With parallel environments, set h to zero for each environment that starts a new episode.
4. **Compare with reference code.** Read CleanRL `ppo_atari_lstm.py` (one file). Read SB3-contrib `RecurrentPPO`. Train an agent in your environment with one of them. If the reference code gives much better results, the problem is in your training code, not in the environment.

---

## 8. Recommendations

Do the tiers in sequence. Do not compare algorithms before you complete Tier 1. If you do, the comparison uses an incorrect benchmark.

### 8.1 Tier 1: correct the environment and the evaluation

1. Do the corrective action for C1 (initial value, penalty, shortfall).
2. Do the corrective action for C2 (stable Hawkes process).
3. Do the corrective action for B1 (seeds).
4. Do the corrective action for C4 (1,000 test seeds, baselines, metrics).
5. Do the corrective action for C3 (2×2 ablation, 5 seeds).

### 8.2 Tier 2: make the training better

1. **Use parallel environments.** Run 32 to 64 environments in each process. Collect N × T steps. Do 4 to 10 epochs on minibatches of sequence chunks. Use truncated BPTT (Section 7.2).
2. **Normalize the value targets**, or use a separate critic (B2).
3. **Set γ = 1.0 or 0.999** (M4).
4. **Make the action space larger.** Use order sizes {-50, -20, -10, 0, +10, +20, +50}, or a continuous size with a Gaussian or Beta policy. A fixed size of 10 shares with a 50-tick latency causes the triangle-wave inventory.
5. **Make the state larger.**
   - Add a histogram of pending fills by remaining time (50 bins). With this feature, the task is fully Markov. It gives a strong MLP baseline.
   - Add the recent realized volatility and the recent event count. With these features, the agent can estimate the Hawkes level. This is the correct reason to use an LSTM.

### 8.3 Tier 3: research extensions, including GRPO

**GRPO (Group Relative Policy Optimization, DeepSeekMath, 2024)**

*How GRPO works*
1. For each "prompt", GRPO collects a group of G complete rollouts.
2. It gives each rollout one scalar reward R_i.
3. It calculates the advantage Â_i = (R_i - mean of the group) / (standard deviation of the group).
4. It uses Â_i for each step of rollout i.
5. It applies the PPO clip objective. Usually it adds a KL penalty to a reference policy.
6. It does not use a critic.

*Advantages for this project*
1. The simulator can start again from the same seed. Thus a "prompt" can be (market seed, initial inventory). The G rollouts then use the same market path (common random numbers). The group mean is then a very good baseline.
2. GRPO optimizes one terminal value directly. A terminal risk measure (entropic risk or CVaR of the terminal PnL) is not a sum of step rewards. A critic cannot learn it easily. GRPO thus corrects M3.
3. GRPO has no critic. Thus it removes B2.

*Disadvantages for this project*
1. All 1,000 actions in an episode receive the same advantage. Thus the credit assignment in time is weak. GAE does this better.
2. GRPO works well for LLMs partly because the policy starts from a strong pretrained model. Your policy starts from random weights.
3. GRPO needs many full episodes for each update. The C++ simulator is fast, so this is possible.

*Recommendation*
1. Use GRPO as a comparison experiment. It is not a guaranteed improvement.
2. Compare two methods on the same test seeds and the same risk metrics:
   - PPO with a reward for each step and GAE
   - GRPO with terminal CVaR or entropic risk and a group baseline with common random numbers
3. A hybrid is also possible: use shorter sub-episodes as group members, or add a small learned value function to the group baseline.

```python
# GRPO update (sketch, no critic)
seed, inv0 = rng.integers(1e9), rng.choice([-500, -250, 250, 500])
group = [rollout(policy, env.reset(seed=seed, inv0=inv0)) for _ in range(G)]  # same market path
R = torch.tensor([-entropic_risk_sample(g.shortfall) for g in group])          # terminal objective
A = (R - R.mean()) / (R.std() + 1e-8)                                         # one value for each episode
loss = -mean_i mean_t min(ratio_it * A_i, clip(ratio_it, 1-eps, 1+eps) * A_i) + beta * KL(pi || pi_ref)
```

**Other methods that fit this task**
1. **Deep Hedging (Buehler et al., thesis reference [2]).**
   - Copy the price and fill logic to PyTorch.
   - Use continuous trade sizes.
   - Minimize the risk measure of the terminal PnL with a direct pathwise gradient. This method does not use RL.
   - With a fast simulator, this method needs few samples. It is also a strong baseline for each RL method.
2. **Distributional PPO or CVaR-PPO.** Keep PPO. Learn the distribution of the return. Optimize CVaR, not the mean.
3. **A more realistic latency model.**
   - Use a random latency (jitter).
   - Use partial fills against the book.
   - Add price impact from the agent orders (M1).
   - Under these conditions, memory is truly necessary.

### 8.4 Recommended form of the main claim

Use a claim that the data can support. For example:

> "Under action latency, a policy with no memory that sees only the settled inventory goes into a limit cycle. The pending-order state removes most of the limit cycle [ablation]. Recurrence gives more value when the fill times or the volatility level are hidden [ablation]. We evaluated each policy on N = 1,000 seeded episodes against a rule baseline and a TWAP baseline."

---

## Appendix A. Script that reproduces C2

```python
import random, statistics

def run(T=1000, seed=0, base_mu=0.02, alpha=0.15, decay=0.98):
    r = random.Random(seed); exc = 0.0; events = []
    for t in range(T):
        exc *= decay; lam = base_mu + exc
        ev = (r.randrange(1000) / 1000.0) < lam
        if ev: exc += alpha
        events.append(ev)
    lock = next((t for t in range(T - 20) if all(events[t:t + 20])), None)
    return sum(events), lock

for alpha in (0.15, 0.012):
    res = [run(seed=s, alpha=alpha) for s in range(2000)]
    locked = [r[1] for r in res if r[1] is not None]
    print(alpha,
          statistics.mean(r[0] for r in res),          # 941 (0.15) / 46 (0.012)
          len(locked) / len(res),                      # 1.0 / 0.0
          statistics.median(locked) if locked else None)  # 49 / None
```

## Appendix B. Glossary

| Term | Definition |
|---|---|
| Ablation | An experiment that removes one component to measure its effect |
| BPTT | Backpropagation Through Time. Backpropagation on a recurrent network that is unrolled over time. |
| Branching ratio | The mean number of new events that one event causes in a Hawkes process. It must be less than 1 for a stable process. |
| CVaR at 5% | The mean of the worst 5% of outcomes |
| GAE | Generalized Advantage Estimation. A weighted sum of 1-step prediction errors. |
| GRPO | Group Relative Policy Optimization. PPO with a group-mean baseline and no critic. |
| Implementation shortfall | Final portfolio value minus initial portfolio value, with both at market price |
| Limit cycle | An oscillation that continues and does not go to a fixed point |
| Lock-in | In this report: the tick after which an event occurs on each tick |
| TWAP | Time-Weighted Average Price. A schedule that trades equal amounts at equal time intervals. |
