#### Implied Volatility Solvers for Black-Scholes option prices ####


import numpy as np
from quant_finance.options.black_scholes_pricing import black_scholes_price
from quant_finance.options.greeks import BS_vega


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
    
    # Input validation    
    if C <= 0:
        raise ValueError("C must be positive")

    if low_sigma <= 0 or high_sigma <= 0:
        raise ValueError("Volatility bounds must be positive")

    if low_sigma >= high_sigma:
        raise ValueError("low_sigma must be smaller than high_sigma")
        
    if tol <= 0: 
        raise ValueError("Tolerance must be positive")
        
    if max_iter <= 0:
        raise ValueError("Maximum iteration must be positive")

    
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


    
## Implied volatility solver with Newton-Raphson Method
# Recall: The derivative of BS with respect to volatility is Vega

def implied_volatility_newton(
    C,
    S,
    K,
    r,
    T,
    q = 0.0,
    option_type = "call",
    initial_guess = 0.15, #initial volatility guess
    tol = 1e-6,
    max_iter = 1000,
):
    
    #Input validation
    if C <= 0:
        raise ValueError("C must be positive")
        
    if initial_guess <= 0:
        raise ValueError("Initial volatility guess must be positive")
        
    if tol <= 0: 
        raise ValueError("Tolerance must be positive")
        
    if max_iter <= 0:
        raise ValueError("Maximum iteration must be positive")

    
    #Initial sigma starts the iteration process
    sigma_i = initial_guess 
    
    #Iteration
    for i in range(max_iter):
        
        #Calculate BS price (f(σ_n)), pricing error & Vega (f'(σ_n))
        BS_price = black_scholes_price(S, K, r, T, sigma = sigma_i, q = q, option_type = option_type)
        
        pricing_error = BS_price - C
        
        #if pricing error below tolerance, stop the iteration
        if abs(pricing_error) < tol:
            return sigma_i
        
        vega = BS_vega(S, K, r, T, sigma = sigma_i, q = q, option_type = option_type)
        
        #Check that vega is not too small for the division
        if abs(vega) < 1e-10:
            raise ValueError("Vega is too small for Newton iteration")
    
        sigma_i = sigma_i - pricing_error / vega 
    
    return sigma_i


#TODO combine both methods in one implied_volatility function