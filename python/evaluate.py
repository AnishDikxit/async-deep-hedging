"""Evaluate policies on fixed test seeds (audit C4).

All policies run on the same seeds, so they see the same market paths and initial inventories.
"""
import os
import json
import argparse
import numpy as np
import torch
from config import Config
from environment import VectorTradingEnvironment
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
        position = envs.holdings + envs.in_flight_exposure
        band = envs.config.tolerance_band
        return np.where(position > band, -1, np.where(position < -band, 1, 0))


class TWAPPolicy:
    """Liquidates the initial inventory in equal orders at equal time intervals."""
    name = "twap"

    def reset(self, num_envs):
        pass

    def act(self, envs, states):
        # The last fill must arrive before the end of the episode.
        horizon = envs.max_steps - envs.config.latency_ticks
        target = envs.initial_holdings * max(0.0, 1.0 - envs.current_step / horizon)
        gap = envs.holdings + envs.in_flight_exposure - target
        return np.where(gap >= ORDER_SIZE, -1, np.where(gap <= -ORDER_SIZE, 1, 0))


def run_episodes(policy, seeds, cfg, batch_size=100):
    """Run one episode for each seed. Return one row of metrics for each episode."""
    rows = []
    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start:start + batch_size]
        envs = VectorTradingEnvironment(len(batch_seeds), cfg)
        states = envs.reset(batch_seeds)
        policy.reset(envs.num_envs)
        sum_sq_inventory = np.zeros(envs.num_envs)
        ticks_outside = np.zeros(envs.num_envs)
        done = False
        while not done:
            with torch.no_grad():
                actions = policy.act(envs, states)
            states, _, done = envs.step(actions)
            holdings = envs.holdings.astype(float)
            sum_sq_inventory += holdings ** 2
            ticks_outside += np.abs(holdings) > cfg.tolerance_band
        shortfall, legacy = envs.implementation_shortfall(), envs.legacy_mtm_pnl()
        for i, seed in enumerate(batch_seeds):
            rows.append({
                "seed": int(seed),
                "initial_holdings": int(envs.initial_holdings[i]),
                "shortfall": float(shortfall[i]),
                "legacy_mtm_pnl": float(legacy[i]),
                "penalty": float(envs.total_penalty[i]),
                "fills": int(envs.num_fills[i]),
                "spread_cost": float(envs.spread_cost[i]),
                "rms_inventory": float(np.sqrt(sum_sq_inventory[i] / envs.max_steps)),
                "pct_ticks_outside_band": float(100.0 * ticks_outside[i] / envs.max_steps),
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
