import torch
import numpy as np
import matplotlib.pyplot as plt

def price_xva(risk_factors):
    """
    Simplified MTM function demonstrating Wrong-Way Risk.
    Uses native PyTorch tensor operations to build the computational graph.
    """
    fx = risk_factors[0]
    spread = risk_factors[1]
    
    # Non-linear exposure logic: torch.relu is equivalent to max(0, x)
    exposure = torch.relu(fx - 1.0) * 1_000_000
    pd = 1.0 - torch.exp(-spread * 2.0) # 2-year probability of default
    
    return exposure * pd

def extract_cross_gamma():
    # Initialize market factors as a PyTorch tensor with gradient tracking
    # Example: FX = 1.05, Credit Spread = 200 bps
    current_market = torch.tensor([1.05, 0.02], requires_grad=True) 
    
    # torch.autograd.functional.hessian calculates the exact 2nd derivative matrix
    gamma_matrix = torch.autograd.functional.hessian(price_xva, current_market)
    
    cross_gamma = gamma_matrix[0, 1].item()
    print("Full Hessian Matrix:\n", gamma_matrix)
    print(f"\nExtracted Cross-Gamma (FX vs Spread): {cross_gamma:.2f}")
    
    return cross_gamma

if __name__ == "__main__":
    extract_cross_gamma()

def run_threshold_optimization(cross_gamma_input):
    # 1. GLOBAL PARAMETERS
    np.random.seed(42)
    N_sim = 5000
    N_DAYS = 250
    dt = 1.0 / 250.0
    
    # Market and Liquidity Parameters
    base_vol = 0.20             # High volatility to exacerbate costs
    Gamma = cross_gamma_input   # Cross-gamma extracted from PyTorch AAD
    alpha = 0.0002              # Base spread (2 bps)
    beta = 0.002                # Spread sensitivity to volatility
    
    # Risk Aversion Parameter 
    lambda_risk = 1e-2

    # Range of D thresholds to test 
    D_values = np.linspace(5000, 50000, 20)
    
    # Arrays to store the U-Curve data
    total_slippage_costs = []
    tracking_error_penalties = []
    objective_function = []

    # 2. PRE-GENERATE ASSET PATHS 
    # We generate one set of paths so all thresholds face the exact same market
    vol_matrix = np.full((N_sim, N_DAYS), base_vol)
    dW = np.random.normal(0, np.sqrt(dt), (N_sim, N_DAYS))
    returns = vol_matrix * dW - 0.5 * (vol_matrix**2) * dt
    S = 1.0 * np.exp(np.cumsum(returns, axis=1))

    # 3. OPTIMIZATION LOOP OVER D
    print("Running Hedging Band Optimization...")
    for D in D_values:
        cost_dynamic_paths = np.zeros((N_sim, N_DAYS))
        unhedged_delta = np.zeros(N_sim)
        daily_unhedged_pnl = np.zeros((N_sim, N_DAYS))
        
        for t in range(1, N_DAYS):
            # Market move drives delta change
            dS = S[:, t] - S[:, t-1]
            unhedged_delta += Gamma * dS
            
            # 1. Track Variance: Mark-to-market of the unhedged position
            daily_unhedged_pnl[:, t] = unhedged_delta * dS
            
            # 2. Rebalancing Mechanics
            rehedge_mask = np.abs(unhedged_delta) >= D
            # You must trade the entire accumulated delta to reset to 0
            trade_amounts = rehedge_mask * np.abs(unhedged_delta) 
            unhedged_delta[rehedge_mask] = 0.0 
            
            # 3. Execution Cost (Crossing the spread)
            spread_dynamic = alpha + beta * vol_matrix[:, t]
            cost_dynamic_paths[:, t] = trade_amounts * (spread_dynamic / 2.0)

        # Calculate metrics for this specific D
        mean_cumulative_cost = np.mean(np.sum(cost_dynamic_paths, axis=1))
        
        path_variances = np.var(daily_unhedged_pnl, axis=1)
        mean_variance = np.mean(path_variances)
        penalty = lambda_risk * mean_variance
        
        # Store results
        total_slippage_costs.append(mean_cumulative_cost)
        tracking_error_penalties.append(penalty)
        objective_function.append(mean_cumulative_cost + penalty)

    # Find the mathematical minimum of the U-Curve
    optimal_idx = np.argmin(objective_function)
    D_star = D_values[optimal_idx]
    min_cost = objective_function[optimal_idx]

    # 4. VISUALIZATION
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the competing forces
    ax.plot(D_values, total_slippage_costs, label='Transaction Costs (Slippage)', color='#1f77b4', linestyle='--', linewidth=2)
    ax.plot(D_values, tracking_error_penalties, label=r'Tracking Error Penalty ($\lambda \cdot Var$)', color='#d62728', linestyle='--', linewidth=2)
    
    # Plot the Objective Function 
    ax.plot(D_values, objective_function, label='Total Objective Function', color='#2ca02c', linewidth=3)
    
    # Highlight the optimal point D*
    ax.plot(D_star, min_cost, marker='o', markersize=8, color='black')
    ax.axvline(D_star, color='black', linestyle=':', alpha=0.6)
    
    ax.set_xlabel(r"Discrete Hedging Threshold ($D$)")
    ax.set_ylabel("Annualized Equivalent Cost (USD)")
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', frameon=True, edgecolor='black', fancybox=False)
    
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 5. LATEX TABLE GENERATOR
    # ==========================================
    print("\n--- COPY AND PASTE THIS LATEX CODE INTO YOUR THESIS ---")
    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\caption{Empirical Optimization of the Hedging Threshold}")
    print("    \\label{tab:optimal_d}")
    print("    \\renewcommand{\\arraystretch}{1.2}")
    print("    \\begin{tabular}{@{}lr@{}}")
    print("        \\toprule")
    print("        \\textbf{Metric} & \\textbf{Value} \\\\")
    print("        \\midrule")
    print(f"        Optimal Threshold ($D^*$) & {int(D_star):,} units \\\\")
    print(f"        Minimized Total Cost & \\${min_cost:,.0f} \\\\")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("\\end{table}")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    run_threshold_optimization(cross_gamma_input=1_000_000)