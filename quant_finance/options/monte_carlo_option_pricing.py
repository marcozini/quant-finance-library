#### Monte Carlo Calculation for European Option Pricing ####

import numpy as np

# Use the solution of Geometric Brownian Motion in the Black-Scholes model as base
#and simulate thourhg Monte Carlo Simulation the option price

# Simulate terminal underlying prices under risk-neutral geometric Brownian motion
def simulate_terminal_prices(S_0, r, sigma, T, q=0, nsim = 10000):
    
    Z = np.random.normal(loc=0, scale=1, size=nsim)
    S_T = S_0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    return S_T


# Price a European call or put option using Monte Carlo simulation.
def monte_carlo_option_price(K, S_0, r, sigma, T, q=0, nsim = 10000, option_type='call'):
    
    S_T = simulate_terminal_prices(S_0, r, sigma, T, q=q, nsim=nsim)
    
    # Average discounted payoff
    if option_type == "call":
        price_i = np.exp(-r * T) * np.maximum(S_T - K, 0)
        price = np.mean(price_i)
    
    elif option_type == "put":
        price_i = np.exp(-r * T) * np.maximum(K - S_T, 0)
        price = np.mean(price_i)
        
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    return price



