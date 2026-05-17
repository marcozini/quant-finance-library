#### Example: Calculate implied volatility from an Black-Scholes option price ####


#import functions
from quant_finance.options.black_scholes_pricing import black_scholes_price
from quant_finance.options.implied_volatility import implied_volatility_bisection

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

# Calculate implied volatility
iv = implied_volatility_bisection(
    C = market_price,
    S = S,
    K = K,
    r = r,
    T = T,
    q = q,
    option_type = option_type,
)

print("Market price:", market_price)
print("True sigma:", sigma_true)
print("Implied volatility:", iv)
print("Difference:", iv - sigma_true)