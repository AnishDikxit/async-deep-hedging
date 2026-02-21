import redis
import time

# 1. Connect to the local Redis server
# decode_responses=True automatically converts byte strings to normal strings
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
pubsub = r.pubsub()

# 2. Subscribe to the C++ market data feed
pubsub.subscribe('market_data')
print("Python AI Agent is online. Listening for market data...")

# 3. The Event Loop (Listening for C++ Broadcasts)
for message in pubsub.listen():
    if message['type'] == 'message':
        
        # Parse the incoming price
        current_price = int(message['data'])
        print(f"[Python] Saw price: ${current_price}")
        
        # 4. The "AI" Decision Logic
        # For this test, our strategy is simple: "Buy the dip!"
        if current_price <= 98:
            print(f"   -> [Python] Price dropped to ${current_price}! Sending BUY order.")
            
            # Format the string exactly how C++ expects it: "PRICE IS_BUY"
            # 1 means True (Buy)
            order_string = f"{current_price} 1"
            
            # 5. Send the order to C++ via the incoming_orders list
            # We use LPUSH (Left Push) so C++ can LPOP it on the other side
            r.lpush('incoming_orders', order_string)
            
            # Sleep for 1 second so we don't spam 10,000 orders at once
            time.sleep(1)