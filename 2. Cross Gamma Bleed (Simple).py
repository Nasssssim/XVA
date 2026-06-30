import torch
import numpy as np
import matplotlib.pyplot as plt

# PyTorch AAD & Pricing Engine

def calculate_cva(S, h, K=100.0, LGD=0.6, T=1.0):
    """Simplified CVA pricer using pure PyTorch tensor operations"""
    exposure = torch.relu(S - K) 
    pd = 1.0 - torch.exp(-h * T)
    cva = LGD * exposure * pd
    return cva

def compute_hybrid_cross_gamma(S_val, h_val, bump=1e-4):
    """Calculates Cross-Gamma (FX/Credit) via Hybrid AAD-FD"""
    # Base Pass
    S = torch.tensor(S_val, requires_grad=True)
    h = torch.tensor(h_val, requires_grad=True)
    cva_base = calculate_cva(S, h)
    cva_base.backward()
    delta_h_base = h.grad.clone()
    
    # Bumped Pass
    S_bumped = torch.tensor(S_val + bump, requires_grad=True)
    h_bumped = torch.tensor(h_val, requires_grad=True)
    cva_bumped = calculate_cva(S_bumped, h_bumped)
    cva_bumped.backward()
    delta_h_bumped = h_bumped.grad.clone()
    
    # Hybrid Cross-Gamma
    cross_gamma = (delta_h_bumped - delta_h_base) / bump
    return cross_gamma.item()

# PART 2: Market Path Generator (with Stress)

def simulate_stressed_paths(S0=105.0, h0=0.05, rho=-0.6, 
                            mu_S=0.0, mu_h=0.0, 
                            num_days=250, num_paths=1):
    """
    Simulates correlated GBM with a volatility spike between day 100 and 150
    """
    dt = 1.0 / 250.0
    S_paths = np.zeros((num_paths, num_days))
    h_paths = np.zeros((num_paths, num_days))
    vol_paths = np.zeros((num_paths, num_days)) # Track vol for spread calculation
    
    S_paths[:, 0] = S0
    h_paths[:, 0] = h0
    
    # Define baseline vols
    base_vol_S = 0.10
    base_vol_h = 0.40
    
    for t in range(1, num_days):
        # Trigger Stress Scenario between day 100 and 150
        if 100 <= t <= 150:
            vol_S = base_vol_S * 3.0
            vol_h = base_vol_h * 3.0
        else:
            vol_S = base_vol_S
            vol_h = base_vol_h
            
        vol_paths[:, t-1] = vol_S # Store prevailing vol for hedging step
        
        Z1 = np.random.standard_normal(num_paths)
        Z2 = np.random.standard_normal(num_paths)
        
        dW_S = Z1 * np.sqrt(dt)
        dW_h = (rho * Z1 + np.sqrt(1.0 - rho**2) * Z2) * np.sqrt(dt)
        
        S_paths[:, t] = S_paths[:, t-1] * np.exp((mu_S - 0.5 * vol_S**2) * dt + vol_S * dW_S)
        h_paths[:, t] = h_paths[:, t-1] * np.exp((mu_h - 0.5 * vol_h**2) * dt + vol_h * dW_h)
        
    return S_paths, h_paths, vol_paths

# PART 3: The Hedging Simulation Engine

def run_path_hedging(fx_path, haz_path, vol_path, threshold_D):
    """
    Runs the discrete hedging algorithm over a single simulated market path
    """
    portfolio_delta = 0.0
    cumulative_cost = 0.0
    daily_costs = np.zeros(len(fx_path))
    
    alpha = 0.0002 # Baseline spread
    beta = 0.001   # Volatility multiplier
    
    # Scale up Gamma artificially
    portfolio_multiplier = 1_000_000 
    
    for t in range(1, len(fx_path)):
        dS = fx_path[t] - fx_path[t-1]
        current_fx = fx_path[t-1]
        current_hazard = haz_path[t-1]
        current_vol = vol_path[t-1]
        
        # 1. Compute exact structural Cross-Gamma (Cross Gamma 1)
        unit_cross_gamma = compute_hybrid_cross_gamma(current_fx, current_hazard)
        total_cross_gamma = unit_cross_gamma * portfolio_multiplier
        
        # 2. Accumulate gamma bleed
        portfolio_delta += total_cross_gamma * dS
        
        # 3. Check Boundary & Execute Hedge (Cross Gamma 2)
        if abs(portfolio_delta) >= threshold_D:
            spread = alpha + (beta * current_vol)
            amount_to_trade = abs(portfolio_delta) - threshold_D 
            
            slippage = amount_to_trade * (spread / 2)
            cumulative_cost += slippage
            
            portfolio_delta = np.sign(portfolio_delta) * threshold_D
            
        daily_costs[t] = cumulative_cost
        
    return daily_costs

# PART 4: Execution & Visualization

if __name__ == "__main__":
    print("Generating market paths...")
    np.random.seed(42)
    num_days = 250
    num_paths = 100
    
    S_sim, h_sim, vol_sim = simulate_stressed_paths(num_paths=num_paths, num_days=num_days)
    
    print("Running hedging simulations across thresholds...")
    thresholds_to_test = [5000, 20000] # Test a tight band vs a loose band
    average_costs_over_time = {D: np.zeros(num_days) for D in thresholds_to_test}
    
    # Loop over thresholds and paths
    for D in thresholds_to_test:
        for p in range(num_paths):
            costs = run_path_hedging(S_sim[p], h_sim[p], vol_sim[p], threshold_D=D)
            average_costs_over_time[D] += costs
        average_costs_over_time[D] /= num_paths
        print(f"Finished Threshold D={D}, Final Avg Cost: ${average_costs_over_time[D][-1]:.2f}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for D in thresholds_to_test:
        plt.plot(average_costs_over_time[D], label=f'Simulated Bleed (D={D})', linewidth=2)
    
    # Highlight the stress period
    plt.axvspan(100, 150, color='red', alpha=0.1, label='Market Stress (Vol x3)')
    

    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Cumulative Slippage Cost ($)", fontsize=12)
    plt.legend(loc="upper left")
    
    plt.grid(False) 
    
    plt.tight_layout()
    
    plt.show()