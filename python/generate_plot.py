import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducible, thesis-quality charts
np.random.seed(42)

# --- Parameters ---
T = 1000  # Number of time steps
dt = 1.0

# --- 1. Simulate Geometric Brownian Motion (GBM) ---
# GBM assumes constant drift and constant volatility
mu_gbm = 0.0
sigma_gbm = 0.012
S_gbm = np.zeros(T)
S_gbm[0] = 100.0

for t in range(1, T):
    dW = np.random.normal(0, np.sqrt(dt))
    S_gbm[t] = S_gbm[t-1] * np.exp((mu_gbm - 0.5 * sigma_gbm**2) * dt + sigma_gbm * dW)

# --- 2. Simulate Hawkes Process (Clustered Volatility) ---
# Hawkes assumes a baseline quiet market, but shocks increase the probability of more shocks
mu_hawkes = 0.01      # Baseline background intensity (quiet market)
alpha = 0.08          # Jump excitation (how much one trade triggers others)
beta = 0.15           # Decay rate (how fast the market calms down)

S_hawkes = np.zeros(T)
S_hawkes[0] = 100.0
lambda_t = mu_hawkes
sigma_base = 0.05     # Quiet baseline volatility

for t in range(1, T):
    # Exponential decay of the intensity function
    lambda_t = mu_hawkes + (lambda_t - mu_hawkes) * np.exp(-beta * dt)
    
    # Determine if a microstructure shock happens based on current intensity
    if np.random.uniform(0, 1) < lambda_t:
        # Shock occurs! Spike the intensity (self-exciting memory)
        lambda_t += alpha
        # Add a heavy-tailed price jump
        shock = np.random.normal(0, 1.8) 
    else:
        # Normal, quiet market tick
        shock = np.random.normal(0, sigma_base)
        
    S_hawkes[t] = S_hawkes[t-1] + shock

# --- 3. Plotting the Comparison ---
# Use a highly professional style
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Top Plot: GBM
ax1.plot(S_gbm, color='#1f77b4', linewidth=1.2) # Deep blue
ax1.set_title('Geometric Brownian Motion (Independent, Constant Volatility)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Asset Price', fontsize=10)
ax1.margins(x=0)

# Bottom Plot: Hawkes
ax2.plot(S_hawkes, color='#d62728', linewidth=1.2) # Crimson red
ax2.set_title('Hawkes Process (Self-Exciting, Clustered Volatility)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Simulation Time Steps (Milliseconds)', fontsize=10)
ax2.set_ylabel('Asset Price', fontsize=10)
ax2.margins(x=0)

# Clean up layout and save in high resolution (300 dpi is standard for print/thesis)
plt.tight_layout()
plt.savefig('volatility_comparison.png', dpi=300, bbox_inches='tight')
print("Successfully generated and saved 'volatility_comparison.png'")
plt.show()