import os
import json
import random
import shutil
import argparse
import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from environment import VectorTradingEnvironment
from agent import PPOAgentLSTM # Using the new LSTM Brain!
from config import Config, TRAIN_SEED_OFFSET
from evaluate import AgentPolicy, run_episodes, summarize

# --- HYPERPARAMETERS ---
# Audit C6: all hyperparameters are in config.py.
# Each run writes them to runs/<run_name>/config.json.

def calculate_entropic_utility(raw_pnl, lambda_val):
    scaled_pnl = raw_pnl / 1000.0
    clipped_pnl = np.clip(scaled_pnl, -100.0, 100.0)
    exponent = np.clip(-lambda_val * clipped_pnl, -40.0, 40.0)
    return float((1.0 - np.exp(exponent)) / lambda_val)

def transform_reward(raw_rewards, cfg):
    if cfg.reward_transform == "entropic_step":
        # Audit M3: this applies the utility to each step reward.
        # The sum of these values is not the entropic risk of the terminal PnL.
        return np.array([calculate_entropic_utility(r, cfg.risk_aversion_lambda) for r in raw_rewards])
    return raw_rewards / cfg.reward_scale

class RunningReturnStd:
    # Audit B2: divide rewards by a running std of the discounted return.
    # A positive scale does not change the optimal policy. It keeps the critic loss near the actor loss.
    def __init__(self, num_envs, gamma):
        self.gamma = gamma
        self.returns = np.zeros(num_envs)
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def reset_returns(self):
        self.returns[:] = 0.0

    def update_and_scale(self, rewards):
        self.returns = self.returns * self.gamma + rewards
        batch_mean, batch_var, n = self.returns.mean(), self.returns.var(), len(self.returns)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total
        return rewards / np.sqrt(self.var + 1e-8)

def worker_process(job):
    worker_id, cfg, run_dir = job
    torch.set_num_threads(1)

    # Audit B1: each core has its own seed. Without this, all cores start with the same weights.
    seed = cfg.base_seed + worker_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_rng = np.random.default_rng(seed)

    # The batched environment steps all simulators in one C++ call.
    envs = VectorTradingEnvironment(cfg.num_envs, cfg)
    agent = PPOAgentLSTM(learning_rate=cfg.learning_rate, hidden_dim=cfg.hidden_dim, use_lstm=cfg.use_lstm)

    weights_path = os.path.join(run_dir, f"weights_core_{worker_id}.pth")
    if os.path.exists(weights_path):
        try: agent.policy_network.load_state_dict(torch.load(weights_path))
        except Exception as error:
            # A checkpoint from another architecture cannot load. Training then starts from new weights.
            print(f"[Core {worker_id}] Could not load {weights_path}: {error}")

    history_pnl = []
    # Diagnostics for each update. They show whether the policy learns to hold and whether the critic fits.
    log_path = os.path.join(run_dir, f"train_log_core_{worker_id}.csv")
    log_file = open(log_path, "w")
    log_file.write("update,episode,shortfall,penalty,fills,hold_fraction,entropy,explained_variance,critic_actor_ratio\n")
    reward_norm = RunningReturnStd(cfg.num_envs, cfg.gamma) if cfg.normalize_value else None
    num_updates = cfg.total_episodes // cfg.num_envs

    for update in range(1, num_updates + 1):
        # Audit B1: record a market seed for each episode, so that each episode can be run again.
        episode_seeds = (TRAIN_SEED_OFFSET + seed_rng.integers(0, 10**9, size=cfg.num_envs)).tolist()
        states = envs.reset(episode_seeds)

        # THE FIX: Wipe the LSTM's memory clean at the start of every episode
        hidden = agent.get_initial_hidden(cfg.num_envs)
        if reward_norm is not None:
            reward_norm.reset_returns()

        buffer_states, buffer_actions, buffer_logprobs = [], [], []
        buffer_rewards, buffer_state_values = [], []

        while True:
            # The AI reads the state AND its own memory
            # Audit B3, B7: num_envs environments run in lockstep and share one forward pass.
            actions, log_probs, _, state_values, hidden = agent.select_action(states, hidden)
            # All episodes have the same length, so all environments finish on the same tick.
            next_states, rewards, done = envs.step(actions)

            utility = transform_reward(rewards, cfg)
            if reward_norm is not None:
                utility = reward_norm.update_and_scale(utility)

            buffer_states.append(states)
            buffer_actions.append(actions + 1)  # network index = action + 1 for ACTION_MAPPING [-1, 0, 1]
            buffer_logprobs.append(log_probs)
            buffer_state_values.append(state_values)
            buffer_rewards.append(utility)

            states = next_states

            # We only perform PPO Math at the EXACT end of an episode
            if done:
                rewards_arr = np.stack(buffer_rewards, axis=1)                   # [N, T]
                values_arr = torch.stack(buffer_state_values, dim=1).numpy()     # [N, T]
                num_steps = rewards_arr.shape[1]

                advantages = np.zeros_like(rewards_arr)
                gae = np.zeros(cfg.num_envs)
                for step in reversed(range(num_steps)):
                    if step == num_steps - 1:
                        # Audit B4: no bootstrap value at the terminal step. The state includes the remaining time.
                        next_val = 0.0
                        is_terminal = 1 # End of the simulation
                    else:
                        next_val = values_arr[:, step + 1]
                        is_terminal = 0 # Middle of the sequence

                    delta = rewards_arr[:, step] + cfg.gamma * next_val * (1 - is_terminal) - values_arr[:, step]
                    gae = delta + cfg.gamma * cfg.gae_lambda * (1 - is_terminal) * gae
                    advantages[:, step] = gae

                # Format exactly for the LSTM: [N Batch, 1000 Sequence Steps, 4 Features]
                old_states = torch.as_tensor(np.stack(buffer_states, axis=1), dtype=torch.float32)
                old_actions = torch.as_tensor(np.stack(buffer_actions, axis=1), dtype=torch.long)
                old_logprobs = torch.stack(buffer_logprobs, dim=1).detach()
                old_state_values = torch.as_tensor(values_arr, dtype=torch.float32)
                advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32)

                returns = advantages_tensor + old_state_values
                # Explained variance near 1 means the critic predicts the returns well. Near 0 or below means it does not.
                returns_np = advantages + values_arr
                explained_variance = 1.0 - np.var(returns_np - values_arr) / (np.var(returns_np) + 1e-8)
                hold_fraction = float(np.mean(np.stack(buffer_actions) == 1))  # network index 1 = hold
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

                # Backpropagate through time
                # Audit B3: each minibatch holds full sequences, because the LSTM needs the time order.
                for _ in range(cfg.ppo_epochs):
                    for mb in torch.randperm(cfg.num_envs).chunk(cfg.num_minibatches):
                        initial_hidden_for_update = agent.get_initial_hidden(len(mb))
                        logprobs, state_values, dist_entropy = agent.evaluate(
                            old_states[mb], old_actions[mb], initial_hidden_for_update, cfg.tbptt_len)

                        ratios = torch.exp(logprobs - old_logprobs[mb])
                        surr1 = ratios * advantages_tensor[mb]
                        surr2 = torch.clamp(ratios, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantages_tensor[mb]

                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(state_values, returns[mb])

                        loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * dist_entropy.mean()

                        agent.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), max_norm=cfg.max_grad_norm)
                        agent.optimizer.step()

                # Audit C1: report the implementation shortfall, not cash + holdings * price - starting cash.
                episode_shortfalls = envs.implementation_shortfall().tolist()
                history_pnl.extend(episode_shortfalls)
                terminal_pnl = float(np.mean(episode_shortfalls))
                mean_penalty = float(envs.total_penalty.mean())
                mean_fills = float(envs.num_fills.mean())
                entropy = float(dist_entropy.mean().item())
                loss_ratio = cfg.value_coef * critic_loss.item() / (abs(actor_loss.item()) + 1e-8)
                break

        episode = update * cfg.num_envs
        log_file.write(f"{update},{episode},{terminal_pnl:.2f},{mean_penalty:.2f},{mean_fills:.1f},"
                       f"{hold_fraction:.4f},{entropy:.4f},{explained_variance:.4f},{loss_ratio:.4f}\n")
        log_file.flush()
        if update % max(1, 500 // cfg.num_envs) == 0:
            torch.save(agent.policy_network.state_dict(), weights_path)
        if update % max(1, 50 // cfg.num_envs) == 0:
            # Audit B2: the loss ratio shows whether the critic gradient is much larger than the actor gradient.
            print(f"[Core {worker_id}] Episode {episode:04d} | Shortfall: ${terminal_pnl:8.2f} | "
                  f"critic/actor loss: {loss_ratio:8.2f} | penalty: {mean_penalty:9.1f} | fills: {mean_fills:5.1f} | "
                  f"hold: {hold_fraction:.2f} | entropy: {entropy:.3f} | expl. var: {explained_variance:5.2f}")

    torch.save(agent.policy_network.state_dict(), weights_path)
    log_file.close()

    # Audit C4: select the winner on separate validation seeds, not on the training PnL.
    agent.policy_network.eval()
    validation = summarize(run_episodes(AgentPolicy(agent), cfg.validation_seeds, cfg))
    return worker_id, seed, float(np.mean(history_pnl[-100:])), validation

def parse_args():
    parser = argparse.ArgumentParser(description="Train independent PPO agents, one for each core.")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--arch", choices=["lstm", "mlp"], default="lstm", help="mlp is the ablation of audit C3")
    parser.add_argument("--no-in-flight", action="store_true", help="hide in_flight_exposure (ablation of audit C3)")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="episodes for each core")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = Config(use_lstm=(args.arch == "lstm"), observe_in_flight=not args.no_in_flight, base_seed=args.base_seed)
    if args.cores is not None:
        cfg.num_cores = args.cores
    if args.episodes is not None:
        cfg.total_episodes = args.episodes

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    mp.set_start_method('spawn')
    # Audit M6: the cores train independent agents. They do not share weights, gradients or data.
    print(f"Launching {cfg.num_cores} independent PPO-{args.arch.upper()} runs (one for each core)...")

    with mp.Pool(processes=cfg.num_cores) as pool:
        results = pool.map(worker_process, [(i, cfg, run_dir) for i in range(cfg.num_cores)])

    print("\n--- All Cores Finished PPO Training ---")
    best_core, best_score, best_seed = -1, -float('inf'), None

    for worker_id, seed, train_score, validation in results:
        # The validation objective is the episode reward sum: shortfall minus the band penalty.
        score = validation["mean_shortfall"] - validation["mean_penalty"]
        print(f"Core {worker_id} (seed {seed}) Train Avg Shortfall (Last 100 Eps): ${train_score:8.2f} | "
              f"Validation objective: {score:8.2f}")
        if score > best_score:
            best_score = score; best_core = worker_id; best_seed = seed

    print(f"\n[*] WINNER: Core {best_core} (Score: ${best_score:8.2f})")

    # Audit C4: copy the winner to the file that test.py and evaluate.py load, and record its origin.
    shutil.copyfile(os.path.join(run_dir, f"weights_core_{best_core}.pth"), "deep_hedging_weights.pth")
    shutil.copyfile(os.path.join(run_dir, "config.json"), "deep_hedging_config.json")
    with open(os.path.join(run_dir, "winner.json"), "w") as f:
        json.dump({"core": best_core, "seed": best_seed, "validation_objective": best_score,
                   "validation_seeds": [cfg.validation_seeds[0], cfg.validation_seeds[-1]],
                   "cores": [{"core": w, "seed": s, "train_shortfall_last100": t, "validation": v}
                             for w, s, t, v in results]}, f, indent=2)
    print(f"Saved deep_hedging_weights.pth and {os.path.join(run_dir, 'winner.json')}")
