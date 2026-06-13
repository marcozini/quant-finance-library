#### Example: Calculate implied volatility from an Black-Scholes option price ####


#import functions
from quant_finance.options.black_scholes_pricing import black_scholes_price
from quant_finance.options.implied_volatility import implied_volatility


# Parameters
S = 100
K = 100
r = 0.01
T = 1
q = 0.0
sigma_true = 0.2
option_type = "call"


# Generate a model price using a known volatility
market_price = black_scholes_price(
    S, K, r, T,
    sigma=sigma_true,
    q=q,
    option_type=option_type,
)

# Calculate implied volatility with bisection
iv_bisection = implied_volatility(
    C = market_price,
    S = S,
    K = K,
    r = r,
    T = T,
    q = q,
    option_type = option_type,
    method = "bisection"
)

print("Market price:", market_price)
print("True sigma:", sigma_true)
print("Implied volatility:", iv_bisection)
print("Difference:", iv_bisection - sigma_true)


# Calculate implied volatility with Newton-Raphson
iv_newton = implied_volatility(
    C = market_price,
    S = S,
    K = K,
    r = r,
    T = T,
    q = q,
    option_type = option_type,
    method = "newton"
)

print("Market price:", market_price)
print("True sigma:", sigma_true)
print("Implied volatility:", iv_newton)
print("Difference:", iv_newton - sigma_true)

