"""Evaluate policies on fixed test seeds (audit C4).

All policies run on the same seeds, so they see the same market paths and initial inventories.
"""
import os
import json
import argparse
import numpy as np
import torch
from config import Config
from environment import TradingEnvironment
from agent import PPOAgentLSTM

ORDER_SIZE = 10


class AgentPolicy:
    """The trained agent with greedy actions."""
    name = "agent"

    def __init__(self, agent):
        self.agent = agent

    def reset(self, num_envs):
        self.hidden = self.agent.get_initial_hidden(num_envs)

    def act(self, envs, states):
        actions, _, _, _, self.hidden = self.agent.select_action(states, self.hidden, test_mode=True)
        return actions


class RulePolicy:
    """Trades toward the band and counts the pending orders (audit C4, item 2)."""
    name = "rule"

    def reset(self, num_envs):
        pass

    def act(self, envs, states):
        actions = []
        for env in envs:
            position = env.holdings + env.in_flight_exposure
            band = env.config.tolerance_band
            actions.append(-1 if position > band else (1 if position < -band else 0))
        return np.array(actions)


class TWAPPolicy:
    """Liquidates the initial inventory in equal orders at equal time intervals."""
    name = "twap"

    def reset(self, num_envs):
        pass

    def act(self, envs, states):
        actions = []
        for env in envs:
            # The last fill must arrive before the end of the episode.
            horizon = env.max_steps - env.config.latency_ticks
            target = env.initial_holdings * max(0.0, 1.0 - env.current_step / horizon)
            gap = env.holdings + env.in_flight_exposure - target
            actions.append(-1 if gap >= ORDER_SIZE else (1 if gap <= -ORDER_SIZE else 0))
        return np.array(actions)


def run_episodes(policy, seeds, cfg, batch_size=100):
    """Run one episode for each seed. Return one row of metrics for each episode."""
    rows = []
    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start:start + batch_size]
        envs = [TradingEnvironment(cfg) for _ in batch_seeds]
        states = np.stack([env.reset(seed=s) for env, s in zip(envs, batch_seeds, strict=True)])
        policy.reset(len(envs))
        sum_sq_inventory = np.zeros(len(envs))
        ticks_outside = np.zeros(len(envs))
        done = False
        while not done:
            with torch.no_grad():
                actions = policy.act(envs, states)
            results = [env.step(int(a)) for env, a in zip(envs, actions, strict=True)]
            states = np.stack([r[0] for r in results])
            done = results[0][2]
            holdings = np.array([env.holdings for env in envs], dtype=float)
            sum_sq_inventory += holdings ** 2
            ticks_outside += np.abs(holdings) > cfg.tolerance_band
        for i, (env, seed) in enumerate(zip(envs, batch_seeds, strict=True)):
            rows.append({
                "seed": int(seed),
                "initial_holdings": env.initial_holdings,
                "shortfall": env.implementation_shortfall(),
                "legacy_mtm_pnl": env.legacy_mtm_pnl(),
                "penalty": env.total_penalty,
                "fills": env.num_fills,
                "spread_cost": env.spread_cost,
                "rms_inventory": float(np.sqrt(sum_sq_inventory[i] / env.max_steps)),
                "pct_ticks_outside_band": float(100.0 * ticks_outside[i] / env.max_steps),
            })
    return rows


def summarize(rows):
    shortfall = np.array([r["shortfall"] for r in rows])
    worst = np.sort(shortfall)[:max(1, int(np.ceil(0.05 * len(shortfall))))]
    return {
        "episodes": len(rows),
        "mean_shortfall": float(shortfall.mean()),
        "std_shortfall": float(shortfall.std(ddof=1)) if len(rows) > 1 else 0.0,
        "cvar5_shortfall": float(worst.mean()),   # mean of the worst 5% of episodes
        "rms_inventory": float(np.sqrt(np.mean([r["rms_inventory"] ** 2 for r in rows]))),
        "pct_ticks_outside_band": float(np.mean([r["pct_ticks_outside_band"] for r in rows])),
        "mean_fills": float(np.mean([r["fills"] for r in rows])),
        "mean_spread_cost": float(np.mean([r["spread_cost"] for r in rows])),
        "mean_penalty": float(np.mean([r["penalty"] for r in rows])),
    }


def load_config(path):
    cfg = Config()
    if path and os.path.exists(path):
        with open(path) as f:
            for key, value in json.load(f).items():
                setattr(cfg, key, value)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies on the fixed test seeds.")
    parser.add_argument("--policies", nargs="+", default=["agent", "rule", "twap"], choices=["agent", "rule", "twap"])
    parser.add_argument("--weights", default="deep_hedging_weights.pth")
    parser.add_argument("--config", default="deep_hedging_config.json")
    parser.add_argument("--num-seeds", type=int, default=None, help="use the first N test seeds (default: all 1,000)")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.test_seeds[:args.num_seeds] if args.num_seeds else cfg.test_seeds
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    for name in args.policies:
        if name == "agent":
            agent = PPOAgentLSTM(hidden_dim=cfg.hidden_dim, use_lstm=cfg.use_lstm)
            agent.policy_network.load_state_dict(torch.load(args.weights))
            agent.policy_network.eval()
            policy = AgentPolicy(agent)
        else:
            policy = RulePolicy() if name == "rule" else TWAPPolicy()
        rows = run_episodes(policy, seeds, cfg)
        summary[name] = summarize(rows)
        with open(os.path.join(args.out, f"eval_{name}.csv"), "w") as f:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r.values()) + "\n")

    summary_path = os.path.join(args.out, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"seeds": [seeds[0], seeds[-1]], "policies": summary}, f, indent=2)

    print(f"{'policy':8s} {'mean IS':>10s} {'std IS':>10s} {'CVaR5 IS':>10s} {'RMS inv':>8s} {'%out':>6s} {'fills':>7s}")
    for name, s in summary.items():
        print(f"{name:8s} {s['mean_shortfall']:10.2f} {s['std_shortfall']:10.2f} {s['cvar5_shortfall']:10.2f} "
              f"{s['rms_inventory']:8.1f} {s['pct_ticks_outside_band']:6.1f} {s['mean_fills']:7.1f}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
