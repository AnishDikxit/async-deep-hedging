"""All hyperparameters for training and evaluation (audit C6).

Each training run writes these values to runs/<run_name>/config.json.
The "thesis code" comments give the value that made the thesis results and Fig. 4.3.
"""
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # --- Market simulator (src/exchange.cpp) ---
    alpha_jump: float = 0.012        # thesis code: 0.15 (branching ratio 7.35, unstable). Audit C2.
    latency_ticks: int = 50          # fixed in exchange.cpp; recorded here for the run log

    # --- Environment (environment.py) ---
    max_steps: int = 1000
    starting_cash: float = 10000.0
    inventory_min: int = -500        # I0 ~ U(-500, 500), integers. thesis code: {-500, -250, 250, 500}. Audit C6.
    inventory_max: int = 500
    tolerance_band: int = 30
    penalty_coef: float = 0.001
    observe_in_flight: bool = True   # False zeroes the in_flight_exposure feature (ablation, audit C3)

    # --- Network (agent.py) ---
    use_lstm: bool = True            # False gives the feed-forward agent with the same hidden size (audit C3)
    hidden_dim: int = 64
    learning_rate: float = 1e-4

    # --- PPO (train_multicore.py) ---
    total_episodes: int = 8000       # episodes for each core
    num_cores: int = 10
    num_envs: int = 16               # environments that run in lockstep in each core (audit B3, B7)
    num_minibatches: int = 4         # each minibatch holds full sequences of num_envs / num_minibatches envs
    ppo_epochs: int = 10             # thesis code: 4. Raised after pilot 1: 16 envs per update gave 4x fewer gradient steps.
    clip_epsilon: float = 0.2
    gamma: float = 0.999             # thesis code: 0.99 (horizon of about 100 ticks). Audit M4.
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01       # thesis code: 0.05. README and Table 3.1 give 0.01. Audit C6.
    max_grad_norm: float = 0.5
    tbptt_len: int = 64              # thesis code: full BPTT over 1,000 steps. Audit C6.

    # --- Reward ---
    reward_transform: str = "linear"  # "linear" or "entropic_step". thesis code: "entropic_step". Audit M3.
    reward_scale: float = 1000.0      # rewards are divided by this value before the transform
    risk_aversion_lambda: float = 0.05  # used only by "entropic_step"
    normalize_value: bool = True      # divide rewards by a running std of the return. thesis code: False. Audit B2.

    # --- Seeds (audit B1, C4) ---
    base_seed: int = 0               # core k uses base_seed + k
    validation_seeds: list = field(default_factory=lambda: list(range(5_000, 5_100)))
    test_seeds: list = field(default_factory=lambda: list(range(0, 1_000)))

    def to_dict(self):
        return asdict(self)


# Training episodes draw their market seeds from this range.
# It does not overlap the validation seeds or the test seeds.
TRAIN_SEED_OFFSET = 1_000_000
