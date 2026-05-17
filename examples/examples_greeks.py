#### Example: Black-Scholes Greeks ####

from quant_finance.options.greeks import (
    BS_delta,
    BS_gamma,
    BS_vega,
    BS_theta,
    BS_rho,
)

# Parameters
S = 100
K = 100
r = 0.01
T = 1
q = 0.0
sigma = 0.2


# Calculate Greeks for call
call_delta = BS_delta(S, K, r, T, sigma, q = q, option_type = "call")
call_gamma = BS_gamma(S, K, r, T, sigma, q = q, option_type = "call")
call_vega = BS_vega(S, K, r, T, sigma, q = q, option_type = "call")
call_theta = BS_theta(S, K, r, T, sigma, q = q, option_type = "call")
call_rho = BS_rho(S, K, r, T, sigma, q = q, option_type = "call")


# Calculate Greeks for a put
put_delta = BS_delta(S, K, r, T, sigma, q = q, option_type = "put")
put_gamma = BS_gamma(S, K, r, T, sigma, q = q, option_type = "put")
put_vega = BS_vega(S, K, r, T, sigma, q = q, option_type = "put")
put_theta = BS_theta(S, K, r, T, sigma, q = q, option_type = "put")
put_rho = BS_rho(S, K, r, T, sigma, q = q, option_type = "put")


print("Call Greeks")
print("Delta:", call_delta)
print("Gamma:", call_gamma)
print("Vega:", call_vega)
print("Theta:", call_theta)
print("Rho:", call_rho)

print("\nPut Greeks")
print("Delta:", put_delta)
print("Gamma:", put_gamma)
print("Vega:", put_vega)
print("Theta:", put_theta)
print("Rho:", put_rho)

