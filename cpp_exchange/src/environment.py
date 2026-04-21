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
        self.max_steps = 1000      # 1 episode = 1000 market ticks
        self.starting_cash = 10000.0
        
        # Tracking variables
        self.current_step = 0
        self.holdings = 0
        self.cash = self.starting_cash
        self.current_price = 100.0

    def reset(self):
        """
        Starts a new trading episode (e.g., a new 30-day contract period).
        Returns the initial State tensor.
        """
        # 1. Flush any leftover delayed orders in the network pipeline
        self.r.delete('incoming_orders')
        
        # 2. Reset the portfolio
        self.current_step = 0
        # AI wakes up with anywhere from -500 (Short) to +500 (Long) shares
        # It MUST trade to neutralize this risk before the Hawkes process kills it
        self.holdings = random.choice([-500, -250, 250, 500])
        self.cash = self.starting_cash
        
        # 3. Wait for the first fresh price tick from C++ to anchor the state
        self.r.rpush("incoming_orders", "RESET")
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
        portfolio_value = self.cash + (self.holdings * self.current_price)
        reward = portfolio_value - self.starting_cash
        
        done = self.current_step >= self.max_steps
        
        return self._get_state(), float(reward), done

    def _get_state(self):
        """
        Packages the environment variables into a standardized NumPy array.
        CRITICAL: Normalizes all inputs to a [-1, 1] or [0, 1] range to prevent 
        PyTorch Softmax saturation.
        """
        time_remaining = self.max_steps - self.current_step
        
        # 1. Normalize Price: Assuming starting price is 100, and typical volatility moves it +/- 10
        norm_price = (self.current_price - 100.0) / 10.0 
        
        # 2. Normalize Holdings: Assuming max position size during exploration could hit 5000
        # This ensures +/- 500 starting exposure enters the brain cleanly as +/- 0.1
        norm_holdings = self.holdings / 5000.0
        
        # 3. Normalize Time: Scale from [1000 -> 0] down to [1.0 -> 0.0]
        norm_time = time_remaining / self.max_steps
        
        # State vector: [Normalized Price, Normalized Inventory, Normalized Time]
        state = np.array([
            norm_price, 
            norm_holdings, 
            norm_time
        ], dtype=np.float32)
        
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