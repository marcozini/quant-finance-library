#### Black-Scholes Examples: put-call parity check & sensitivity plots ####

import numpy as np
import matplotlib.pyplot as plt

from quant_finance.options.black_scholes_pricing import black_scholes_price


## put-call parity check (C - P = S - K*exp(-rT))

S = 100
K = 95
r = 0.01
T = 1
sigma = 0.2

call_price = black_scholes_price(S, K, r, T, sigma, q = 0, option_type = "call")
put_price  = black_scholes_price(S, K, r, T, sigma, q = 0, option_type = "put")

diff = (call_price - put_price) - (S - K * np.exp(-r*T))

print(call_price)
print(put_price)
print(diff)
print(np.isclose(diff, 0, atol = 1e-6))


## Sensitivity analysis

# 1) Price vs S
S_range = np.linspace(50, 150, 100)
K = 100
r = 0.01
T = 1
sigma = 0.2

call_prices = [black_scholes_price(s, K, r, T, sigma, q = 0, option_type = "call") for s in S_range]
put_prices  = [black_scholes_price(s, K, r, T, sigma, q = 0, option_type = "put")  for s in S_range]

plt.figure(figsize=(9, 5))
plt.plot(S_range, call_prices, label="Call", color="blue")
plt.plot(S_range, put_prices,  label="Put",  color="red")
plt.axvline(x=K, color="gray", linestyle="--", label="Strike (K)")
plt.xlabel("Underlying Price (S)")
plt.ylabel("Option Price")
plt.title("Black-Scholes Price vs Underlying Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 2) Price vs T
T_range = np.linspace(0.01, 2, 200)
S = 100
K = 100
r = 0.01
sigma = 0.2

call_prices = [black_scholes_price(S, K, r, t, sigma, option_type = "call") for t in T_range]
put_prices  = [black_scholes_price(S, K, r, t, sigma, option_type = "put")  for t in T_range]

plt.figure(figsize=(9, 5))
plt.plot(T_range, call_prices, label="Call", color="blue")
plt.plot(T_range, put_prices,  label="Put",  color="red")
plt.xlabel("Time-to-Maturity (T)")
plt.ylabel("Option Price")
plt.title("Black-Scholes Price vs Time-to-Maturity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 3) Price vs sigma
sigma_range = np.linspace(0.01, 1.0, 200)
S = 100
K = 100      
r = 0.01
T = 1

call_prices = [black_scholes_price(S, K, r, T, s, option_type = "call") for s in sigma_range]
put_prices  = [black_scholes_price(S, K, r, T, s, option_type = "put")  for s in sigma_range]

plt.figure(figsize=(9, 5))
plt.plot(sigma_range, call_prices, label="Call", color="blue")
plt.plot(sigma_range, put_prices,  label="Put",  color="red")
plt.xlabel("Volatility (sigma)")
plt.ylabel("Option Price")
plt.title("Black-Scholes Price vs Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 4) Price vs risk-free interest rate
r_range = np.linspace(-0.02, 0.10, 200)  
S = 100
K = 100
T = 1
sigma = 0.2

call_prices = [black_scholes_price(S, K, r, T, sigma, option_type = "call") for r in r_range]
put_prices  = [black_scholes_price(S, K, r, T, sigma, option_type = "put")  for r in r_range]

plt.figure(figsize=(9, 5))
plt.plot(r_range, call_prices, label="Call", color="blue")
plt.plot(r_range, put_prices,  label="Put",  color="red")
plt.axvline(x=0, color="gray", linestyle="--", label="r = 0")
plt.xlabel("Risk-Free Rate (r)")
plt.ylabel("Option Price")
plt.title("Black-Scholes Price vs Interest Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 5) Price vs strike
K_range = np.linspace(50, 150, 200)
S = 100
r = 0.01
T = 1
sigma = 0.2

call_prices = [black_scholes_price(S, k, r, T, sigma, option_type = "call") for k in K_range]
put_prices  = [black_scholes_price(S, k, r, T, sigma, option_type = "put")  for k in K_range]

plt.figure(figsize=(9, 5))
plt.plot(K_range, call_prices, label="Call", color="blue")
plt.plot(K_range, put_prices,  label="Put",  color="red")
plt.axvline(x=S, color="gray", linestyle="--", label="Spot (S)")
plt.xlabel("Strike Price (K)")
plt.ylabel("Option Price")
plt.title("Black-Scholes Price vs Strike Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






