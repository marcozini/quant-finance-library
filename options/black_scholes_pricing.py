#### Black-Scholes Pricing ####

import numpy as np
from scipy import stats


# C = Call option price
# P = Put option price
# S = Current underlying price
# K = Strike price
# r = Risk-free interest rate
# T = Time to maturity
# sigma = standard deviation of the underlying asset
# N = Normal distribution
# q = annual dividend yield (continuously compounded)

def black_scholes_price(S, K, r, T, sigma, q = 0.0, option_type = "call"):
    
    # Input validation
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if q < 0:
        raise ValueError("Dividend yield can not be negative")
        
    #Calculation
    d1 = (np.log(S/K) + (r - q + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call": 
        C = S * np.exp(-q*T) * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
        return C
    
    elif option_type == "put":
        P = K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * np.exp(-q*T) * stats.norm.cdf(-d1)
        return P
    
    else: 
        raise ValueError("option_type must be 'call' or 'put'")
   



