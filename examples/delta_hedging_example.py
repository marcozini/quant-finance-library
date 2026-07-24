#### Delta-Hedging Example ####

import numpy as np
import matplotlib.pyplot as plt

from quant_finance.options.delta_hedging import delta_hedge


#parameters
S_0 = 100
K = 100
r = 0.05
T = 1.0
sigma = 0.20
q = 0.0
N = 252 #trading days
seed = 0
option_type = "call"
position = "short"


##Simulate one delta-hedged option position
results = delta_hedge(
    S_0=S_0,
    K=K,
    r=r,
    T=T,
    sigma=sigma,
    N=N,
    q=q,
    seed=seed,
    option_type=option_type,
    position=position,
)


#extract the results
S_path = results["S_path"]
option_prices = results["option_prices"]
delta_path = results["delta_path"]
stock_positions = results["stock_positions"]
cash_account = results["cash_account"]
payoff = results["payoff"]
terminal_pnl = results["terminal_pnl"]


#construct the time grids
time_grid = np.linspace(0.0, T, N + 1)
hedging_times = time_grid[:-1]


#calculate values useful for analysis
position_sign = 1 if position == "long" else -1

hedge_portfolio_value = stock_positions * S_path[:-1] + cash_account
total_hedged_value = position_sign * option_prices + hedge_portfolio_value


#summary
print(f"Initial option price:       {option_prices[0]:.4f}")
print(f"Terminal stock price:       {S_path[-1]:.4f}")
print(f"Option payoff:              {payoff:.4f}")
print(f"Terminal hedging P&L:       {terminal_pnl:.4f}")
print(f"Stock-path length:          {len(S_path)}")
print(f"Option-price-path length:   {len(option_prices)}")
print(f"Delta-path length:          {len(delta_path)}")
print(f"Stock-position-path length: {len(stock_positions)}")
print(f"Cash-account-path length:   {len(cash_account)}")


#visualization
opposite_signed_option_value = -position_sign * option_prices

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

#stock-price path and strike
axes[0, 0].plot(time_grid, S_path, label="Stock price")
axes[0, 0].axhline(K, color="red", linestyle="--", label="Strike")
axes[0, 0].set_title("Stock-Price Path")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Price")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

#Black-Scholes delta and stock position
axes[0, 1].plot(hedging_times, delta_path, label="Black-Scholes delta")
axes[0, 1].plot(hedging_times, stock_positions, linestyle="--", label="Stock position")
axes[0, 1].set_title("Delta and Stock Position")
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Number of shares")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

#cash-account balance
axes[1, 0].plot(hedging_times, cash_account, color="green")
axes[1, 0].set_title("Cash-Account Balance")
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Cash balance")
axes[1, 0].grid(alpha=0.3)

#hedge portfolio versus opposite signed option value
axes[1, 1].plot(hedging_times, hedge_portfolio_value, label="Hedging portfolio")
axes[1, 1].plot(hedging_times, opposite_signed_option_value, linestyle="--", label="Opposite signed option value")
axes[1, 1].set_title("Hedge Portfolio and Option Value")
axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("Value")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


#compare different rebalancing frequencies
frequencies = [12, 52, 252]
n_simulations = 1000

freq_sim = np.empty((len(frequencies), n_simulations))

mean_pnl = np.empty(len(frequencies))
std_pnl = np.empty(len(frequencies))
mae_pnl = np.empty(len(frequencies))
rmse_pnl = np.empty(len(frequencies))


for i, freq in enumerate(frequencies):

    terminal_pnl_sim = np.empty(n_simulations)

    for nsim in range(n_simulations):

        results = delta_hedge(
            S_0=S_0,
            K=K,
            r=r,
            T=T,
            sigma=sigma,
            N=freq,
            q=q,
            seed=seed + nsim,
            option_type=option_type,
            position=position,
        )

        terminal_pnl_sim[nsim] = results["terminal_pnl"]

    # Store all terminal P&Ls for this frequency
    freq_sim[i] = terminal_pnl_sim

    # Calculate summary statistics
    mean_pnl[i] = np.mean(terminal_pnl_sim)
    std_pnl[i] = np.std(terminal_pnl_sim, ddof=1)
    mae_pnl[i] = np.mean(np.abs(terminal_pnl_sim))
    rmse_pnl[i] = np.sqrt(np.mean(terminal_pnl_sim**2))


# Print the results
for i, freq in enumerate(frequencies):

    print(f"\nRebalancing frequency:                  N = {freq}")
    print(f"Mean P&L:                               {mean_pnl[i]:.4f}")
    print(f"Standard deviation:                     {std_pnl[i]:.4f}")
    print(f"MAE (Mean absolute error):              {mae_pnl[i]:.4f}")
    print(f"RMSE (root mean square deviation):      {rmse_pnl[i]:.4f}")


# Plot the frequency comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))


# Boxplot of terminal P&L by rebalancing frequency
axes[0].boxplot(
    [freq_sim[i] for i in range(len(frequencies))]
)

axes[0].set_xticks(range(1, len(frequencies) + 1))
axes[0].set_xticklabels(frequencies)
axes[0].axhline(0.0, color="black", linestyle="--")
axes[0].set_title("Terminal Hedging P&L")
axes[0].set_xlabel("Rebalancing Frequency N")
axes[0].set_ylabel("Terminal P&L")
axes[0].grid(alpha=0.3)


# Standard deviation, MAE and RMSE against N
axes[1].plot(
    frequencies,
    std_pnl,
    marker="o",
    label="Standard deviation",
)

axes[1].plot(
    frequencies,
    mae_pnl,
    marker="o",
    label="MAE",
)

axes[1].plot(
    frequencies,
    rmse_pnl,
    marker="o",
    label="RMSE",
)

axes[1].set_title("Hedging Error by Rebalancing Frequency")
axes[1].set_xlabel("Rebalancing Frequency N")
axes[1].set_ylabel("Error measure")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


##Conclusion
# More frequent rebalancing generally reduces the dispersion of the
# terminal hedging P&L. Consequently, the standard deviation, MAE and
# RMSE should decrease as N increases, while the mean P&L should remain
# close to zero.

# However, this relationship holds across many simulated paths.
# For an individual path, more frequent rebalancing does not guarantee
# a smaller absolute hedging error because of random path behaviour.

# These results assume the Black-Scholes model, correct model parameters,
# no transaction costs and no dividend yield.
