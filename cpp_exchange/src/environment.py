import redis
import numpy as np

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
        self.holdings = 0
        self.cash = self.starting_cash
        
        # 3. Wait for the first fresh price tick from C++ to anchor the state
        self.r.rpush("incoming_orders", "RESET")
        self.current_price = self._wait_for_next_tick()
        
        return self._get_state()

    def step(self, action):
        """
        Executes the AI's chosen action, advances time by 1 tick, and calculates the reward.
        Returns: (next_state, reward, done)
        """
        self.current_step += 1
        
        # --- 1. EXECUTE THE ACTION ---
        # Action Mapping: 1 (Buy), -1 (Sell), 0 (Hold)
        if action == 1:
            # Format: "PRICE IS_BUY" (e.g., "102 1")
            order_str = f"{int(self.current_price)} 1"
            self.r.lpush('incoming_orders', order_str)
            
            # Assume instant fill for local accounting (C++ handles actual latency slippage)
            self.holdings += 10
            self.cash -= (self.current_price * 10)
            
        elif action == -1:
            order_str = f"{int(self.current_price)} 0"
            self.r.lpush('incoming_orders', order_str)
            
            self.holdings -= 10
            self.cash += (self.current_price * 10)

        # --- 2. ADVANCE THE CLOCK ---
        # Block and wait for the C++ engine to broadcast the next price movement
        next_price = self._wait_for_next_tick()
        self.current_price = next_price
        
        # --- 3. CALCULATE REWARD ---
        # Mark-to-Market Portfolio Value
        portfolio_value = self.cash + (self.holdings * self.current_price)
        
        # The reward is the net profit/loss since the start of the episode
        reward = portfolio_value - self.starting_cash
        
        # --- 4. CHECK TERMINATION ---
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
        
        # 2. Normalize Holdings: Assuming max position size is roughly 100 shares
        norm_holdings = self.holdings / 100.0
        
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