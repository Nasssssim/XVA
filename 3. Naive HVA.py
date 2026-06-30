import numpy as np
import matplotlib.pyplot as plt

def simulate_scenario(base_vol, N_sim, N_DAYS, dt, S0, Gamma, D, alpha, beta, crisis_start, crisis_end, crisis_multiplier):
    """
    Monte Carlo
    """
    # 1. Market Profile
    vol_profile = np.full(N_DAYS, base_vol)
    vol_profile[crisis_start:crisis_end] *= crisis_multiplier
    vol_matrix = np.tile(vol_profile, (N_sim, 1))
    
    # 2. Geometric Brownian Motion paths
    dW = np.random.normal(0, np.sqrt(dt), (N_sim, N_DAYS))
    returns = vol_matrix * dW - 0.5 * (vol_matrix**2) * dt
    S = S0 * np.exp(np.cumsum(returns, axis=1))
    
    # 3. Discrete Hedging Engine
    cost_naive_paths = np.zeros((N_sim, N_DAYS))
    cost_dynamic_paths = np.zeros((N_sim, N_DAYS))
    unhedged_delta = np.zeros(N_sim)
    
    for t in range(1, N_DAYS):
        dS = S[:, t] - S[:, t-1]
        unhedged_delta += Gamma * dS
        
        rehedge_mask = np.abs(unhedged_delta) >= D
        trade_amounts = rehedge_mask * D
        unhedged_delta[rehedge_mask] = 0.0 
        
        spread_naive = alpha
        spread_dynamic = alpha + beta * vol_matrix[:, t]
        
        cost_naive_paths[:, t] = trade_amounts * (spread_naive / 2.0)
        cost_dynamic_paths[:, t] = trade_amounts * (spread_dynamic / 2.0)

    cum_cost_naive_sim = np.cumsum(np.mean(cost_naive_paths, axis=0))
    cum_cost_dynamic_sim = np.cumsum(np.mean(cost_dynamic_paths, axis=0))

    # 4. Theoretical HVA
    theoretical_naive_daily = ((Gamma**2) / (2 * D)) * (alpha * (vol_profile**2)) * dt
    theoretical_dynamic_daily = ((Gamma**2) / (2 * D)) * (alpha * (vol_profile**2) + beta * (vol_profile**3)) * dt
    
    cum_cost_naive_theo = np.cumsum(theoretical_naive_daily)
    cum_cost_dynamic_theo = np.cumsum(theoretical_dynamic_daily)
    
    return vol_profile, cum_cost_naive_sim, cum_cost_dynamic_sim, cum_cost_naive_theo, cum_cost_dynamic_theo

def run_multi_vol_simulation():
    # 1. GLOBAL PARAMETERS
    np.random.seed(42)
    
    N_sim = 10000       # Standardized simulation path variable
    N_DAYS = 250
    dt = 1.0 / 250.0  
    S0 = 1.0          
    
    # Crisis Parameters
    crisis_start = 100
    crisis_end = 150
    crisis_multiplier = 3.0  
    
    # Portfolio & Liquidity Parameters
    Gamma = 1_000_000   
    D = 20_000          
    alpha = 0.0002      
    beta = 0.002        
    
    vol_levels = [0.10, 0.20, 0.30]
    
    # 2. VISUALIZATION CONFIGURATION
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 3. EXECUTION & PLOTTING LOOP
    for i, base_vol in enumerate(vol_levels):
        ax = axes[i]
        
        # Run simulation for the specific volatility regime
        vol_profile, cum_naive_sim, cum_dyn_sim, cum_naive_theo, cum_dyn_theo = simulate_scenario(
            base_vol, N_sim, N_DAYS, dt, S0, Gamma, D, alpha, beta, crisis_start, crisis_end, crisis_multiplier
        )
        
        # Highlight the Crisis Period
        ax.axvspan(crisis_start, crisis_end, color='#d3d3d3', alpha=0.4, label='Crisis Period')
        
        # Plot Simulated Results
        ax.plot(cum_naive_sim, label='Naive HVA - Simulated', color='#1f77b4', linewidth=1.5)
        ax.plot(cum_dyn_sim, label='True Cost - Simulated', color='#d62728', linewidth=1.5)
        
        # Plot Theoretical Overlays
        ax.plot(cum_naive_theo, label=r'Naive HVA - Theo ($\sigma^2$)', color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.plot(cum_dyn_theo, label=r'True Cost - Theo ($\sigma^3$)', color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel("Trading Days")
        if i == 0:
            ax.set_ylabel("Cumulative Rehedging Cost (USD)")
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0:
            ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_multi_vol_simulation()