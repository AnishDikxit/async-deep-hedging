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
        self.orders_placed = 0
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
        if action != 0:
            self.orders_placed += 1
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

    def pending_cost(self):
        # Half-spread and fee of the orders that are sent but not filled. Each order costs 0.50 * 10 + 1 = $6.
        return 6.0 * self.orders_placed - self.spread_cost

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


class VectorTradingEnvironment:
    """N environments in lockstep on one cpp_exchange.MarketBatch.

    The rules are the same as TradingEnvironment. With the same seeds and actions,
    each environment gives the same states and rewards (tests/test_vector_env.py).
    """
    ORDER_SIZE = 10
    TICKET_FEE = 1.00
    HALF_SPREAD = 0.50

    def __init__(self, num_envs, config=None):
        self.config = config if config is not None else Config()
        self.num_envs = num_envs
        self.batch = cpp_exchange.MarketBatch(num_envs, alpha_jump=self.config.alpha_jump)
        self.max_steps = self.config.max_steps
        self.starting_cash = self.config.starting_cash
        self.current_step = 0

    def reset(self, seeds):
        seeds = [int(s) for s in seeds]
        self.current_step = 0
        # Same inventory draw as TradingEnvironment.reset(seed).
        self.holdings = np.array([int(np.random.default_rng(s).integers(self.config.inventory_min, self.config.inventory_max + 1))
                                  for s in seeds], dtype=np.int64)
        self.initial_holdings = self.holdings.copy()
        self.in_flight_exposure = np.zeros(self.num_envs, dtype=np.int64)
        self.cash = np.full(self.num_envs, self.starting_cash)
        self.num_fills = np.zeros(self.num_envs, dtype=np.int64)
        self.orders_placed = np.zeros(self.num_envs, dtype=np.int64)
        self.spread_cost = np.zeros(self.num_envs)
        self.total_penalty = np.zeros(self.num_envs)

        self.current_price = self.batch.reset([s & 0xFFFFFFFF for s in seeds]).astype(np.float64)
        # Audit C1: the portfolio value includes the initial inventory at market price.
        self.previous_portfolio_value = self.cash + self.holdings * self.current_price
        self.initial_mtm = self.previous_portfolio_value.copy()
        return self._get_state()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int32)
        self.current_step += 1
        self.in_flight_exposure += self.ORDER_SIZE * actions
        self.orders_placed += np.abs(actions)

        # The engine ignores the order price (audit C6). It is passed only to keep the interface.
        price, buy_fills, sell_fills, buy_qty, sell_qty, buy_value, sell_value = self.batch.step(
            actions, self.current_price.astype(np.int32))
        self.current_price = price.astype(np.float64)

        # Settlement: the same cash formula as TradingEnvironment, summed over the fills of the tick.
        self.holdings += buy_qty - sell_qty
        self.in_flight_exposure -= buy_qty - sell_qty
        self.cash -= buy_value + self.HALF_SPREAD * buy_qty + self.TICKET_FEE * buy_fills
        self.cash += sell_value - self.HALF_SPREAD * sell_qty - self.TICKET_FEE * sell_fills
        fills = buy_fills + sell_fills
        self.num_fills += fills
        self.spread_cost += self.HALF_SPREAD * (buy_qty + sell_qty) + self.TICKET_FEE * fills

        # Tolerance-band penalty: reward only, not cash (audit C1).
        excess = np.maximum(0, np.abs(self.holdings) - self.config.tolerance_band)
        inventory_bleed = self.config.penalty_coef * excess.astype(np.float64) ** 2
        self.total_penalty += inventory_bleed

        portfolio_value = self.cash + self.holdings * self.current_price
        rewards = (portfolio_value - self.previous_portfolio_value) - inventory_bleed
        self.previous_portfolio_value = portfolio_value

        done = self.current_step >= self.max_steps
        return self._get_state(), rewards, done

    def implementation_shortfall(self):
        return self.cash + self.holdings * self.current_price - self.initial_mtm

    def pending_cost(self):
        # Half-spread and fee of the orders that are sent but not filled.
        order_cost = self.HALF_SPREAD * self.ORDER_SIZE + self.TICKET_FEE
        return order_cost * self.orders_placed - self.spread_cost

    def legacy_mtm_pnl(self):
        return self.cash + self.holdings * self.current_price - self.starting_cash

    def _get_state(self):
        time_remaining = self.max_steps - self.current_step
        norm_price = np.clip((self.current_price - 100.0) / 10.0, -5.0, 5.0)
        norm_holdings = np.clip(self.holdings / 500.0, -5.0, 5.0)
        norm_time = np.full(self.num_envs, time_remaining / self.max_steps)
        norm_in_flight = np.clip(self.in_flight_exposure / 500.0, -5.0, 5.0)
        if not self.config.observe_in_flight:
            norm_in_flight = np.zeros(self.num_envs)
        return np.stack([norm_price, norm_holdings, norm_time, norm_in_flight], axis=1).astype(np.float32)
