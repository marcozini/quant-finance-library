#### Monte Carlo Calculation for European Option Pricing ####

import numpy as np
from statistics import NormalDist


# Simulate terminal underlying prices under risk-neutral geometric Brownian motion.

def simulate_terminal_prices(S_0, r, sigma, T, q=0, nsim=100000, seed=None):
    
    rng = np.random.default_rng(seed)
    Z = rng.normal(loc=0, scale=1, size=nsim)

    S_T = S_0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    return S_T


# Price a European call or put option using Monte Carlo simulation.
def monte_carlo_option_price(
    K,
    S_0,
    r,
    sigma,
    T,
    q=0,
    nsim=100000,
    option_type="call",
    alpha=0.05,
    seed=None,
    return_stats=False
):
    
    S_T = simulate_terminal_prices(S_0, r, sigma, T, q=q, nsim=nsim, seed=seed)
    
    # Average discounted payoff
    if option_type == "call":
        discounted_payoffs = np.exp(-r * T) * np.maximum(S_T - K, 0)

    elif option_type == "put":
        discounted_payoffs = np.exp(-r * T) * np.maximum(K - S_T, 0)
    
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    price = np.mean(discounted_payoffs)
    
    # Standard error and Z-Confidence Intervals
    standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(nsim)
    
    z = NormalDist().inv_cdf(1 - alpha / 2)
    ci_lower = price - z * standard_error
    ci_upper = price + z * standard_error
    
    if return_stats:
        return price, standard_error, (ci_lower, ci_upper)

    return price
    
    
