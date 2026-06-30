import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# We use EURUSD
ticker = "EURUSD=X" 
start_date = "2019-01-01"
end_date = "2024-01-01"

# "Normal" market spread is 1bp
# "Stressed" market spread expands based on volatility
BASE_SPREAD = 0.0001  
VOL_MULTIPLIER = 0.02 # Sensitivity of spread to vol

print(f"Downloading data for {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Clean up data 
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
    
data = data[['Close']].copy()
data['Returns'] = data['Close'].pct_change()

# We look at the standard deviation of returns over the last 20 days (approx 1 month)
# annualized by sqrt(252)
window = 20
data['Rolling_Vol'] = data['Returns'].rolling(window=window).std() * np.sqrt(252)

# MODEL THE DYNAMIC SPREAD (CROSS GAMMA 2 COST)
# This simulates liquidity drying up during stress events
data['Modelled_Spread'] = BASE_SPREAD + (VOL_MULTIPLIER * data['Rolling_Vol'])

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot 1: The FX Rate 
ax1.set_xlabel('Date')
ax1.set_ylabel('EURUSD Rate', color='tab:blue')
ax1.plot(data.index, data['Close'], color='tab:blue', alpha=0.6, label='EURUSD Spot')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot 2: The Cost of Hedging 
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('modeled Bid-Ask Spread (bps)', color='tab:red')
# Convert to basis points for display 
ax2.plot(data.index, data['Modelled_Spread']*10000, color='tab:red', label='Modeled Spread (Cost)')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.show()

threshold = data['Rolling_Vol'].quantile(0.90) # Top 10% volatility days
stress_cost = data[data['Rolling_Vol'] > threshold]['Modelled_Spread'].mean()
calm_cost = data[data['Rolling_Vol'] <= threshold]['Modelled_Spread'].mean()

print(f"Average Spread (Normal Market): {calm_cost*10000:.2f} bps")
print(f"Average Spread (Stressed Market): {stress_cost*10000:.2f} bps")
print(f"Factor Increase: {stress_cost/calm_cost:.1f}x higher cost during stress")