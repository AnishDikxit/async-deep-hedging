import numpy as np
import random
import cpp_exchange # The newly compiled v2.0 engine!

class TradingEnvironment:
    def __init__(self):
        self.engine = cpp_exchange.MarketSimulator()
        self.max_steps = 1000      
        self.starting_cash = 10000.0
        
        self.current_step = 0
        self.holdings = 0
        self.cash = self.starting_cash
        self.current_price = 100.0
        self.previous_portfolio_value = self.starting_cash

    def reset(self):
        self.engine.reset()
        self.current_step = 0
        self.holdings = random.choice([-500, -250, 250, 500])
        self.cash = self.starting_cash
        self.previous_portfolio_value = self.starting_cash 
        
        # v2.0 UPDATE: step() now returns a tuple (price, fills). 
        # We ignore fills on reset because no orders exist yet.
        self.current_price, _ = self.engine.step()
        
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        ticket_fee = 1.00 
        
        # --- 1. SEND ORDER TO C++ ENGINE (NO INSTANT GRATIFICATION) ---
        if action == 1:
            self.engine.place_order(int(self.current_price), True)
            # CRITICAL REMOVAL: We no longer update holdings or cash here!
            
        elif action == -1:
            self.engine.place_order(int(self.current_price), False)
            # CRITICAL REMOVAL: We no longer update holdings or cash here!

        # --- 2. ADVANCE THE CLOCK AND RECEIVE THE LEDGER ---
        self.current_price, executed_fills = self.engine.step()
        
        # --- 3. THE SETTLEMENT LAYER (DELAYED PNL & SLIPPAGE) ---
        # We only update our portfolio when C++ explicitly confirms the order survived the queue
        for fill in executed_fills:
            if fill.is_buy:
                self.holdings += fill.quantity
                # Notice we use fill.execution_price (the price 50ms later), NOT self.current_price
                self.cash -= (((fill.execution_price + 0.50) * fill.quantity) + ticket_fee)
            else:
                self.holdings -= fill.quantity
                self.cash += (((fill.execution_price - 0.50) * fill.quantity) - ticket_fee)
        
        # --- 4. INVENTORY PENALTY (TOLERANCE BAND) ---
        if abs(self.holdings) <= 30:
            inventory_bleed = 0.0
        else:
            excess_exposure = abs(self.holdings) - 30
            inventory_bleed = 0.001 * (excess_exposure ** 2)
            
        self.cash -= inventory_bleed
        
        # --- 5. CALCULATE DENSE REWARD ---
        portfolio_value = self.cash + (self.holdings * self.current_price)
        step_reward = portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = portfolio_value
        
        done = self.current_step >= self.max_steps
        
        return self._get_state(), float(step_reward), done

    def _get_state(self):
        time_remaining = self.max_steps - self.current_step
        norm_price = np.clip((self.current_price - 100.0) / 10.0, -5.0, 5.0)
        norm_holdings = np.clip(self.holdings / 500.0, -5.0, 5.0)
        norm_time = time_remaining / self.max_steps
        
        return np.array([norm_price, norm_holdings, norm_time], dtype=np.float32)