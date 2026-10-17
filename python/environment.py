import numpy as np
import cpp_exchange # The newly compiled v2.0 engine!
from config import Config

class TradingEnvironment:
    def __init__(self, config=None, seed=None):
        self.config = config if config is not None else Config()
        # Audit B1: the engine and the inventory draw use their own seeded generators.
        self.rng = np.random.default_rng(seed)
        engine_seed = 0 if seed is None else int(seed) & 0xFFFFFFFF
        self.engine = cpp_exchange.MarketSimulator(seed=engine_seed, alpha_jump=self.config.alpha_jump)
        self.max_steps = self.config.max_steps
        self.starting_cash = self.config.starting_cash

        self.current_step = 0
        self.holdings = 0
        self.cash = self.starting_cash
        self.current_price = 100.0
        self.previous_portfolio_value = self.starting_cash
        self.initial_mtm = self.starting_cash

    def reset(self, seed=None):
        # A seed makes the episode reproducible (same market path and same initial inventory).
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.engine.reset(seed=int(seed) & 0xFFFFFFFF)
        else:
            self.engine.reset()
        self.current_step = 0
        # Audit C6: I0 ~ U(-500, 500) as the README states. The thesis code used random.choice([-500, -250, 250, 500]).
        self.holdings = int(self.rng.integers(self.config.inventory_min, self.config.inventory_max + 1))
        self.initial_holdings = self.holdings
        self.in_flight_exposure = 0  # NEW: Track pending shares
        self.cash = self.starting_cash

        # Audit C5: counters for the evaluation report.
        self.num_fills = 0
        self.spread_cost = 0.0       # half-spread and ticket fees paid
        self.total_penalty = 0.0     # sum of the tolerance-band penalty (reward only, not cash)

        # v2.0 UPDATE: step() now returns a tuple (price, fills).
        # We ignore fills on reset because no orders exist yet.
        self.current_price, _ = self.engine.step()

        # Audit C1: the portfolio value must include the initial inventory at market price.
        # Otherwise the first reward is holdings * price, which the agent did not cause.
        self.previous_portfolio_value = self.cash + self.holdings * self.current_price
        self.initial_mtm = self.previous_portfolio_value

        return self._get_state()

    def step(self, action):
        self.current_step += 1
        ticket_fee = 1.00

        # --- 1. SEND ORDER TO C++ ENGINE (NO INSTANT GRATIFICATION) ---
        # Audit C6: the engine ignores the price argument. All agent orders are market orders.
        if action == 1:
            self.engine.place_order(int(self.current_price), True)
            self.in_flight_exposure += 10 # Log the pending buy
            # CRITICAL REMOVAL: We no longer update holdings or cash here!

        elif action == -1:
            self.engine.place_order(int(self.current_price), False)
            self.in_flight_exposure -= 10 # Log the pending sell
            # CRITICAL REMOVAL: We no longer update holdings or cash here!

        # --- 2. ADVANCE THE CLOCK AND RECEIVE THE LEDGER ---
        self.current_price, executed_fills = self.engine.step()

        # --- 3. THE SETTLEMENT LAYER (DELAYED PNL & SLIPPAGE) ---
        # We only update our portfolio when C++ explicitly confirms the order survived the queue
        for fill in executed_fills:
            self.num_fills += 1
            self.spread_cost += 0.50 * fill.quantity + ticket_fee
            if fill.is_buy:
                self.holdings += fill.quantity
                self.in_flight_exposure -= fill.quantity # Remove from pending
                # Notice we use fill.execution_price (the price 50ms later), NOT self.current_price
                self.cash -= (((fill.execution_price + 0.50) * fill.quantity) + ticket_fee)
            else:
                self.holdings -= fill.quantity
                self.in_flight_exposure += fill.quantity # Remove from pending
                self.cash += (((fill.execution_price - 0.50) * fill.quantity) - ticket_fee)

        # --- 4. INVENTORY PENALTY (TOLERANCE BAND) ---
        band = self.config.tolerance_band
        if abs(self.holdings) <= band:
            inventory_bleed = 0.0
        else:
            excess_exposure = abs(self.holdings) - band
            inventory_bleed = self.config.penalty_coef * (excess_exposure ** 2)

        # Audit C1: the penalty is a training signal, not a cash flow.
        # It goes into the reward only, so the reported PnL contains only market gains, losses and costs.
        self.total_penalty += inventory_bleed

        # --- 5. CALCULATE DENSE REWARD ---
        portfolio_value = self.cash + (self.holdings * self.current_price)
        step_reward = (portfolio_value - self.previous_portfolio_value) - inventory_bleed
        self.previous_portfolio_value = portfolio_value

        done = self.current_step >= self.max_steps

        return self._get_state(), float(step_reward), done

    def implementation_shortfall(self):
        # Audit C1: final MTM minus initial MTM. It does not include the value of the initial inventory.
        return self.cash + self.holdings * self.current_price - self.initial_mtm

    def legacy_mtm_pnl(self):
        # The thesis value: cash + holdings * price - starting cash. It includes the initial inventory value.
        return self.cash + self.holdings * self.current_price - self.starting_cash

    def _get_state(self):
        time_remaining = self.max_steps - self.current_step
        norm_price = np.clip((self.current_price - 100.0) / 10.0, -5.0, 5.0)
        norm_holdings = np.clip(self.holdings / 500.0, -5.0, 5.0)
        norm_time = time_remaining / self.max_steps
        # NEW: Normalize the pending exposure
        norm_in_flight = np.clip(self.in_flight_exposure / 500.0, -5.0, 5.0)
        # Audit C3: the ablation without E_t sets this feature to zero. The network shape stays the same.
        if not self.config.observe_in_flight:
            norm_in_flight = 0.0
        # Return a 4-feature state
        return np.array([norm_price, norm_holdings, norm_time, norm_in_flight], dtype=np.float32)
