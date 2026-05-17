#### Implied Volatility Solvers for Black-Scholes option prices ####


import numpy as np
from quant_finance.options.black_scholes_pricing import black_scholes_price


## Notation

# C = Call option price
# P = Put option price
# S = Current underlying price
# K = Strike price
# r = Risk-free interest rate
# T = Time to maturity
# sigma = standard deviation of the underlying asset
# N = Normal distribution
# q = annual dividend yield (continuously compounded)



def implied_volatility_bisection(
    C,
    S,
    K,
    r,
    T,
    q = 0.0,
    option_type = "call",
    low_sigma = 1e-6,
    high_sigma = 3.0,
    tol = 1e-6,
    max_iter = 1000,
):
    
    # Input aalidation    
    if C <= 0:
        raise ValueError("C must be positive")

    if low_sigma <= 0 or high_sigma <= 0:
        raise ValueError("Volatility bounds must be positive")

    if low_sigma >= high_sigma:
        raise ValueError("low_sigma must be smaller than high_sigma")
    
    # Difference between model and market price; root problem function
    f_low = black_scholes_price(S, K, r, T, sigma = low_sigma, q = q, option_type = option_type) - C
    f_high = black_scholes_price(S, K, r, T, sigma = high_sigma, q = q, option_type = option_type) - C
    
    if np.sign(f_low) == np.sign(f_high):
        raise ValueError("No implied volatility found in the interval [low_sigma, high_sigma]. "
                         "Try increasing high_sigma or check the market price."
    )
    
    #start the iterating
    for i in range(max_iter):
        
        #mid_point volatility for interval update
        mid_sigma = (low_sigma + high_sigma) / 2
        
        #difference between model and market price at midpoint volatility
        f_mid = black_scholes_price(S, K, r, T, sigma = mid_sigma, q = q, option_type = option_type) - C
        
        if abs(f_mid) < tol:
            return mid_sigma
        
        elif np.sign(f_low) == np.sign(f_mid):
            low_sigma = mid_sigma
            f_low = f_mid
        
        else:
            high_sigma = mid_sigma
    
    return mid_sigma


    

    
#TODO: implement Newton's method as well & add them into one function

