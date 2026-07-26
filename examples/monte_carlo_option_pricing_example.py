#### Monte Carlo Option Pricing Example ####

import matplotlib.pyplot as plt
import numpy as np

from quant_finance.options.black_scholes_pricing import black_scholes_price
from quant_finance.options.monte_carlo_option_pricing import monte_carlo_option_price

# Option and market parameters
S_0 = 100
K = 100
r = 0.05
sigma = 0.20
T = 1
q = 0.0
option_type = "call"

# Monte Carlo parameters
nsim = 100_000
seed=0


## Call option pricing and Black-Scholes validation

mc_price, standard_error, confidence_interval = monte_carlo_option_price(
    K=K,
    S_0=S_0,
    r=r,
    sigma=sigma,
    T=T,
    q=q,
    nsim=nsim,
    option_type=option_type,
    alpha=0.05,
    seed=seed,
    return_stats=True,
)

print("Monte Carlo price:", mc_price)
print("Standard error:", standard_error)
print("95% confidence interval:", confidence_interval)

bs_price = black_scholes_price(
    S=S_0,
    K=K,
    r=r,
    T=T,
    sigma=sigma,
    q=q,
    option_type=option_type,
)

print("Black-Scholes price:", bs_price)

pricing_difference = mc_price - bs_price
absolute_error = abs(pricing_difference)

print("Pricing difference between MC & BS:", pricing_difference)
print("Absolute Error between MC & BS:", absolute_error)

ci_lower, ci_upper = confidence_interval
print(
    "Black-Scholes price inside 95% confidence interval:",
    ci_lower <= bs_price <= ci_upper,
)


## Repeated runs illustrate Monte Carlo sampling variability

print("\nRepeated Monte Carlo runs:")
for run in range(1, 6):
    mc_price_run = monte_carlo_option_price(
        K=K,
        S_0=S_0,
        r=r,
        sigma=sigma,
        T=T,
        q=q,
        nsim=nsim,
        seed=seed,
        option_type=option_type,
    )

    pricing_difference_run = mc_price_run - bs_price

    print(
        f"Run {run}: "
        f"MC price = {mc_price_run:.4f}, "
        f"difference = {pricing_difference_run:.4f}"
    )


## Repeated runs illustrate Monte Carlo sampling variability
# Error does not have to decrease monotonically because each point uses
# a different random sample.

nsim_values = np.logspace(
    3,
    6,
    num=20,
    dtype=int,
)


mc_prices = []
absolute_errors = []

for nsim_i in nsim_values:
    mc_price_i = monte_carlo_option_price(
        K=K,
        S_0=S_0,
        r=r,
        sigma=sigma,
        T=T,
        q=q,
        nsim=nsim_i,
        option_type=option_type,
    )

    pricing_difference_i = mc_price_i - bs_price
    absolute_error_i = abs(pricing_difference_i)

    mc_prices.append(mc_price_i)
    absolute_errors.append(absolute_error_i)
    
    
# Plot Monte Carlo price convergence
plt.figure(figsize=(9, 5))

plt.plot(
    nsim_values,
    mc_prices,
    marker="o",
    label="Monte Carlo price",
)

plt.axhline(
    y=bs_price,
    linestyle="--",
    label="Black-Scholes price",
)

plt.xscale("log")
plt.xlabel("Number of simulations")
plt.ylabel("Option price")
plt.title("Monte Carlo convergence to Black-Scholes price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot absolute pricing error
plt.figure(figsize=(9, 5))

plt.plot(
    nsim_values,
    absolute_errors,
    marker="o",
)

plt.xscale("log")
plt.xlabel("Number of simulations")
plt.ylabel("Absolute pricing error")
plt.title("Monte Carlo absolute pricing error")
plt.grid(True)
plt.tight_layout()
plt.show()



## Put option pricing and Black-Scholes validation

put_mc_price, put_standard_error, put_confidence_interval = (
    monte_carlo_option_price(
        K=K,
        S_0=S_0,
        r=r,
        sigma=sigma,
        T=T,
        q=q,
        nsim=nsim,
        option_type="put",
        alpha=0.05,
        return_stats=True,
    )
)

put_bs_price = black_scholes_price(
    S=S_0,
    K=K,
    r=r,
    T=T,
    sigma=sigma,
    q=q,
    option_type="put",
)

put_pricing_difference = put_mc_price - put_bs_price
put_absolute_error = abs(put_pricing_difference)

put_ci_lower, put_ci_upper = put_confidence_interval

print("\nEuropean put validation:")
print("Monte Carlo put price:", put_mc_price)
print("Black-Scholes put price:", put_bs_price)
print("Standard error:", put_standard_error)
print("95% confidence interval:", put_confidence_interval)
print("Pricing difference:", put_pricing_difference)
print("Absolute pricing error:", put_absolute_error)
print(
    "Black-Scholes price inside 95% confidence interval:",
    put_ci_lower <= put_bs_price <= put_ci_upper,
)


    
    
    
    
    