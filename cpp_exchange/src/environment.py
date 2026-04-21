import redis
import numpy as np
import random

class TradingEnvironment:
    def __init__(self):
        """
        Initializes the connection to the C++ Market Simulator via Redis.
        """
        # Connect to the local Redis instance
        self.r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.pubsub = self.r.pubsub()
        self.pubsub.subscribe('market_data')
        
        # Episode parameters
        self.max_steps = 250      # 1 episode = 1000 market ticks
        self.starting_cash = 10000.0
        
        # Tracking variables
        self.current_step = 0
        self.holdings = 0
        self.cash = self.starting_cash
        self.current_price = 100.0
        self.previous_portfolio_value = self.starting_cash # NEW

    def reset(self):
        # 1. Flush delayed orders
        self.r.delete('incoming_orders')
        
        # 2. AGGRESSIVELY flush the Python PubSub buffer of any stale prices
        while True:
            msg = self.pubsub.get_message(ignore_subscribe_messages=True)
            if msg is None:
                break
        
        # 3. Reset the portfolio
        self.current_step = 0
        self.holdings = random.choice([-500, -250, 250, 500])
        self.cash = self.starting_cash
        self.previous_portfolio_value = self.starting_cash 
        
        # 4. Tell C++ to reset
        self.r.rpush("incoming_orders", "RESET")
        
        # 5. Wait for the FIRST fresh price
        self.current_price = self._wait_for_next_tick()
        
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        
        # --- 1. EXECUTE THE ACTION (With Slippage) ---
        if action == 1:
            order_str = f"{int(self.current_price)} 1"
            self.r.lpush('incoming_orders', order_str)
            
            self.holdings += 10
            # PENALTY: Add $0.50 slippage/spread to the purchase price
            self.cash -= ((self.current_price + 0.50) * 10) 
            
        elif action == -1:
            order_str = f"{int(self.current_price)} 0"
            self.r.lpush('incoming_orders', order_str)
            
            self.holdings -= 10
            # PENALTY: Subtract $0.50 slippage/spread from the sell price
            self.cash += ((self.current_price - 0.50) * 10)

        # --- 2. ADVANCE THE CLOCK ---
        self.current_price = self._wait_for_next_tick()
        
        # --- 3. INVENTORY PENALTY (The Deep Hedging Core) ---
        # The AI must bleed cash every single tick it holds an unhedged position
        # A 500 share position squares to 250,000, multiplied by 0.001 = $250 bleed per tick
        inventory_bleed = 0.001 * (self.holdings ** 2)
        self.cash -= inventory_bleed
        
        # --- 4. CALCULATE REWARD ---
        # --- 4. CALCULATE DENSE REWARD ---
        portfolio_value = self.cash + (self.holdings * self.current_price)
        
        # Reward is exactly how much PnL was gained/lost ON THIS SPECIFIC TICK
        step_reward = portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = portfolio_value
        
        done = self.current_step >= self.max_steps
        
        return self._get_state(), float(step_reward), done

    def _get_state(self):
        time_remaining = self.max_steps - self.current_step
        
        # 1. Normalize Price (WITH A CLIPPING SHIELD)
        # np.clip ensures that even if a stray 180 slips through, the AI never sees a number bigger than 5.0
        norm_price = (self.current_price - 100.0) / 10.0 
        norm_price = np.clip(norm_price, -5.0, 5.0)
        
        # 2. Normalize Holdings (AMPLIFY THE SIGNAL)
        # Changing the denominator to 500.0. Now, an exposure of -500 sends a LOUD -1.0 signal to the AI.
        norm_holdings = self.holdings / 500.0
        norm_holdings = np.clip(norm_holdings, -5.0, 5.0)
        
        # 3. Normalize Time
        norm_time = time_remaining / self.max_steps
        
        state = np.array([norm_price, norm_holdings, norm_time], dtype=np.float32)
        return state

    def _wait_for_next_tick(self):
        """
        Internal networking helper. Listens to the Redis bus until 
        the C++ engine publishes a new price.
        """
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                return float(message['data'])


# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    import time
    import random
    
    print("Initializing Market Environment...")
    env = TradingEnvironment()
    
    print("Resetting Environment for Episode 1...")
    state = env.reset()
    print(f"Initial State: {state}")
    
    print("\n--- Starting Random Agent Test ---")
    total_reward = 0
    
    # Run a quick 10-step simulation
    for i in range(10):
        # AI selects a random action: -1, 0, or 1
        random_action = random.choice([-1, 0, 1])
        
        print(f"\nStep {i+1}: Agent Action -> {random_action}")
        
        # Step the environment
        next_state, reward, done = env.step(random_action)
        
        print(f"Next State: {next_state}")
        print(f"Current PnL: ${reward:.2f}")
        
        time.sleep(0.5) # Slow down for terminal readability
        
    print("\nEnvironment networking test complete. Ready for PyTorch integration.")