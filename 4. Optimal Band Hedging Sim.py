import torch
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

#1. SETTINGS & SYNTHETIC DATA GENERATION
N_PATHS = 5000
N_STEPS = 252       # Daily steps for 1 year 
DT = 1.0 / N_STEPS
NOTIONAL = 100_000_000 # Scale up the portfolio to see real slippage dollars

# Market Parameters
S0 = torch.tensor(1.20, requires_grad=True)  # EUR/USD Spot
h0 = torch.tensor(0.05, requires_grad=True)  # Base Hazard Rate 500bp
r_d, r_f = 0.02, 0.01                        # Domestic/Foreign Rates
vol_S = 0.15                                 # FX Vol

# CIR Credit Parameters
kappa_h = 0.5   # Speed of mean reversion
theta_h = 0.05  # Long-term mean hazard rate
vol_h = 0.10    # Volatility of the hazard rate

# WWR: Positive correlation 
rho = 0.6       
LGD = 0.60      # Loss Given Default

def simulate_paths(S_init, h_init, Z1=None, Z2=None):
    # Expand initial scalars to the path dimension
    S_t = S_init.expand(N_PATHS)
    h_t = h_init.expand(N_PATHS)
    
    S_list = [S_t]
    h_list = [h_t]
    
    Z1 = torch.randn((N_PATHS, N_STEPS))
    Z2 = torch.randn((N_PATHS, N_STEPS))
        
    W_S = Z1 * np.sqrt(DT)
    W_h = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(DT)
    
    for t in range(N_STEPS):
        # 1. Calculate next step for S (GBM)
        S_next = S_t * torch.exp((r_d - r_f - 0.5 * vol_S**2) * DT + vol_S * W_S[:, t])
        
        # 2. Calculate next step for h (CIR)
        h_t_clamped = torch.clamp(h_t, min=1e-6) 
        h_next = h_t_clamped + kappa_h * (theta_h - h_t_clamped) * DT + vol_h * torch.sqrt(h_t_clamped) * W_h[:, t]
        
        # Clamp the output before appending
        h_next = torch.clamp(h_next, min=1e-6)
        
        # 3. Append to lists 
        S_list.append(S_next)
        h_list.append(h_next)
        
        # 4. Update current state
        S_t = S_next
        h_t = h_next
        
    # Stack the lists along the time dimension to create the final tensors
    # Resulting shape: (N_PATHS, N_STEPS + 1)
    S = torch.stack(S_list, dim=1)
    h = torch.stack(h_list, dim=1)
    
    return S, h

# 2. THE DIFFERENTIABLE PRICER
def compute_cva(S_init, h_init, Z1=None, Z2=None):
    
    S, h = simulate_paths(S_init, h_init, Z1, Z2)
    exposure = torch.clamp(S - S0.item(), min=0.0) * NOTIONAL # Apply Notional
    
    time_grid = torch.linspace(0, 1, N_STEPS + 1)
    df = torch.exp(-r_d * time_grid)
    
    surv_prob = torch.exp(-torch.cumsum(h * DT, dim=1))
    pd = torch.zeros_like(surv_prob)
    pd[:, 1:] = surv_prob[:, :-1] - surv_prob[:, 1:]
    
    discounted_exposure = exposure * df
    cva_paths = LGD * torch.sum(discounted_exposure * pd, dim=1)
    
    return torch.mean(cva_paths)

# --- 3. HYBRID AAD-FD CROSS-GAMMA CALCULATION ---
print("--- Calculating Greeks ---")

#Generate the random universes ONCE before pricing
Z1_shared = torch.randn((N_PATHS, N_STEPS))
Z2_shared = torch.randn((N_PATHS, N_STEPS))

# Pass the shared matrices into the base calculation
cva_base = compute_cva(S0, h0, Z1=Z1_shared, Z2=Z2_shared)
delta_S, delta_h = torch.autograd.grad(cva_base, (S0, h0), retain_graph=True)

bump = 1e-4
S_bumped = torch.tensor(S0.item() + bump, requires_grad=True)

# Pass THE EXACT SAME matrices into the bumped calculation
cva_bumped = compute_cva(S_bumped, h0, Z1=Z1_shared, Z2=Z2_shared)
delta_S_bumped, delta_h_bumped = torch.autograd.grad(cva_bumped, (S_bumped, h0))

gamma_S = (delta_S_bumped - delta_S) / bump
cross_gamma_Sh = (delta_h_bumped - delta_h) / bump

print(f"Base CVA: ${cva_base.item():,.0f}")
print(f"Hybrid Single Gamma: {gamma_S.item():,.0f}")
print(f"Hybrid Cross-Gamma: {cross_gamma_Sh.item():,.0f}\n")

# 4. HEDGING BANDS & SLIPPAGE BACKTEST
print("--- Running Hedging Simulation ---")

bps_cost = 0.0002  # 2 bps bid-ask spread
risk_aversion = 1.0

# Fix Band
optimal_band = 10e4

# Note: The theoretical cubic root boundary is heuristically scaled here to fit 
# the nominal delta limits of the simulated portfolio for practical backtesting.
optimal_band_scaled = abs(gamma_S.item()) * 0.02 # Allow 2% drift relative to total Gamma
print(f"Optimal No-Trade Band Width: +/- {optimal_band_scaled:,.0f} units of Delta")

# Simulate a single realized market path for the backtest
S_realized, _ = simulate_paths(S0, h0)
S_path = S_realized[0, :].detach().numpy() # Take the first path

def simulate_hedging(path, strategy='continuous', band=0.0):
    portfolio_delta = 0.0
    cumulative_cost = 0.0
    delta_history = []
    cost_history = []
    
    for t in range(len(path) - 1):
        # Unhedged delta drift driven by gamma and market move (dS)
        delta_drift = gamma_S.item() * (path[t+1] - path[t])
        portfolio_delta += delta_drift
        
        trade_qty = 0.0
        if strategy == 'continuous':
            # Rebalance to exactly 0 every step
            trade_qty = -portfolio_delta
            portfolio_delta = 0.0
            
        elif strategy == 'band':
            # Rebalance only if boundary is breached
            if portfolio_delta > band:
                trade_qty = -(portfolio_delta - band)
                portfolio_delta = band
            elif portfolio_delta < -band:
                trade_qty = -(portfolio_delta + band)
                portfolio_delta = -band
                
        # Accrue slippage (Cost = Trade Amount * Price * Spread)
        # We multiply by path[t] because bps cost applies to the asset value
        trade_cost = abs(trade_qty) * path[t] * bps_cost
        cumulative_cost += trade_cost
        
        delta_history.append(portfolio_delta)
        cost_history.append(cumulative_cost)
        
    return cumulative_cost, delta_history, cost_history

cost_continuous, delta_cont, cost_hist_cont = simulate_hedging(S_path, strategy='continuous')
cost_band, delta_band, cost_hist_band = simulate_hedging(S_path, strategy='band', band=optimal_band_scaled)

print(f"Total Slippage (Naive Continuous Hedging): ${cost_continuous:,.2f}")
print(f"Total Slippage (Optimal Band Hedging): ${cost_band:,.2f}")
print(f"Cost Reduction: {(1 - cost_band/cost_continuous)*100:.1f}%\n")

# 5. VISUALIZATIONS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Portfolio Delta vs. Optimal Bands
days = np.arange(len(delta_band))
ax1.plot(days, delta_band, color='tab:blue', alpha=0.8, label='Portfolio Net Delta (Band Strategy)')
ax1.plot(days, delta_cont, color='tab:gray', alpha=0.5, label='Portfolio Net Delta (Continuous)')
ax1.axhline(optimal_band_scaled, color='tab:red', linestyle='--', label='+ No-Trade Boundary')
ax1.axhline(-optimal_band_scaled, color='tab:red', linestyle='--', label='- No-Trade Boundary')
ax1.axhline(0, color='black', linewidth=0.5)

ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Unhedged Delta Exposure')
ax1.legend(loc='upper right')
ax1.grid(False)

# Plot 2: Cumulative Slippage Cost Comparison
ax2.plot(days, cost_hist_cont, color='tab:gray', label=f'Continuous Hedging (${cost_continuous:,.0f})')
ax2.plot(days, cost_hist_band, color='tab:green', linewidth=2, label=f'Optimal Band Hedging (${cost_band:,.0f})')

ax2.set_xlabel('Trading Days')
ax2.set_ylabel('Cumulative Transaction Costs ($)')
ax2.legend(loc='upper left')
ax2.grid(False)

print(optimal_band)
plt.tight_layout()
plt.savefig("hedging_backtest_results.pdf", bbox_inches='tight', format='pdf')
plt.show()
