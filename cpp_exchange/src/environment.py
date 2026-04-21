import numpy as np
import random
import cpp_exchange # Your compiled C++ engine!

class TradingEnvironment:
    def __init__(self):
        # Instantiate the C++ object directly in memory
        self.engine = cpp_exchange.MarketSimulator()
        
        self.max_steps = 1000      
        self.starting_cash = 10000.0
        
        self.current_step = 0
        self.holdings = 0
        self.cash = self.starting_cash
        self.current_price = 100.0
        self.previous_portfolio_value = self.starting_cash

    def reset(self):
        # 1. Reset the C++ backend instantly
        self.engine.reset()
        
        # 2. Reset the Python portfolio
        self.current_step = 0
        self.holdings = random.choice([-500, -250, 250, 500])
        self.cash = self.starting_cash
        self.previous_portfolio_value = self.starting_cash 
        
        # 3. Pull the starting price
        self.current_price = self.engine.step()
        
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        
        # --- 1. SEND ORDER TO C++ ENGINE ---
        if action == 1:
            self.engine.place_order(int(self.current_price), True)
            self.holdings += 10
            self.cash -= ((self.current_price + 0.50) * 10) 
            
        elif action == -1:
            self.engine.place_order(int(self.current_price), False)
            self.holdings -= 10
            self.cash += ((self.current_price - 0.50) * 10)

        # --- 2. ADVANCE THE C++ CLOCK ---
        # This instantly runs the C++ loop and returns the new price
        self.current_price = self.engine.step()
        
        # --- 3. INVENTORY PENALTY ---
        inventory_bleed = 0.001 * (self.holdings ** 2)
        self.cash -= inventory_bleed
        
        # --- 4. CALCULATE DENSE REWARD ---
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