#### Examples Binomial Tree Option Pricing ####

import matplotlib.pyplot as plt

from quant_finance.options.binomial_tree import (
    EuropeanOption,
    AmericanOption,
    BermudanOption,
)
from quant_finance.options.black_scholes_pricing import black_scholes_price


# Common parameters
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.20
q = 0.0


## 1. Small European call tree

small_option = EuropeanOption(
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    N=3,
    q=q,
    option_type="call",
)

small_price, option_tree = small_option.price()
stock_tree = small_option.build_tree()

print("Small European call tree")
print("Option price:", small_price)

for step, values in enumerate(stock_tree):
    print(f"Stock prices at step {step}: {values}")

for step, values in enumerate(option_tree):
    print(f"Option values at step {step}: {values}")


## Visualize the small stock-price tree

plt.figure(figsize=(9, 6))

for step, stock_prices in enumerate(stock_tree):
    for node, stock_price in enumerate(stock_prices):
        plt.scatter(step, stock_price)

        plt.annotate(
            f"{stock_price:.2f}",
            (step, stock_price),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )

        if step < small_option.N:
            plt.plot(
                [step, step + 1],
                [stock_price, stock_tree[step + 1][node]],
            )
            plt.plot(
                [step, step + 1],
                [stock_price, stock_tree[step + 1][node + 1]],
            )

plt.title("Three-Step Binomial Stock-Price Tree")
plt.xlabel("Time step")
plt.ylabel("Stock price")
plt.xticks(range(small_option.N + 1))
plt.tight_layout()
plt.show()


#### 2. European call convergence toward Black-Scholes

black_scholes_call = black_scholes_price(
    S=S0,
    K=K,
    r=r,
    T=T,
    sigma=sigma,
    q=q,
    option_type="call",
)

step_values = [2, 5, 10, 25, 50, 100, 200, 500]
binomial_prices = []

for N in step_values:
    option = EuropeanOption(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        N=N,
        q=q,
        option_type="call",
    )

    price, _ = option.price()
    binomial_prices.append(price)

print("\nEuropean call convergence")

for N, price in zip(step_values, binomial_prices):
    print(f"N={N}: binomial price={price:.6f}")

print(f"Black-Scholes price: {black_scholes_call:.6f}")

plt.figure(figsize=(9, 5))
plt.plot(
    step_values,
    binomial_prices,
    marker="o",
    label="Binomial price",
)
plt.axhline(
    black_scholes_call,
    linestyle="--",
    label="Black-Scholes price",
)

plt.title("Binomial Convergence Toward Black-Scholes")
plt.xlabel("Number of time steps")
plt.ylabel("European call price")
plt.legend()
plt.tight_layout()
plt.show()

print("As the number of time steps increases, the binomial price "
    "converges toward the Black-Scholes price.")


#### 3. European versus American call

european_call = EuropeanOption(
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    N=500,
    q=q,
    option_type="call",
)

american_call = AmericanOption(
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    N=500,
    q=q,
    option_type="call",
)

european_call_price, _ = european_call.price()
american_call_price, _ = american_call.price()

print("\nEuropean and American calls")
print("European call:", european_call_price)
print("American call:", american_call_price)

print("Without dividends, early exercise of an American call is not optimal, "
    "so both prices should be approximately equal.")


#### 4. European, Bermudan, and American puts

european_put = EuropeanOption(
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    N=500,
    q=q,
    option_type="put",
)

bermudan_put = BermudanOption(
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    N=500,
    q=q,
    option_type="put",
    exercise_steps={100, 200, 300, 400},
)

american_put = AmericanOption(
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    N=500,
    q=q,
    option_type="put",
)

european_put_price, _ = european_put.price()
bermudan_put_price, _ = bermudan_put.price()
american_put_price, _ = american_put.price()

print("\nEuropean, Bermudan, and American puts")
print("European put:", european_put_price)
print("Bermudan put:", bermudan_put_price)
print("American put:", american_put_price)

print("The option value increases with exercise flexibility: "
    "European <= Bermudan <= American.")

#### 5. Compare put prices

exercise_styles = ["European", "Bermudan", "American"]
put_prices = [
    european_put_price,
    bermudan_put_price,
    american_put_price,
]

plt.figure(figsize=(7, 5))
plt.bar(exercise_styles, put_prices)

plt.title("Put Value by Exercise Style")
plt.ylabel("Option price")
plt.tight_layout()
plt.show()