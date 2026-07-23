#### Delta-Hedging Simulation ####


import numpy as np

from quant_finance.options.black_scholes_pricing import black_scholes_price
from quant_finance.options.greeks import BS_delta


def simulate_gbm_path(
        S_0,
        T,
        sigma,
        N,
        r,
        q=0.0,
        seed=0,
):
    
    dt = T / N

    dW = np.random.default_rng(seed).normal(0.0, np.sqrt(dt), N)

    S_path = np.empty(N + 1)
    S_path[0] = S_0

    for i in range(1, N + 1):
       S_path[i] = S_path[i - 1] * np.exp(
           (r - q - 0.5 * sigma**2) * dt + sigma * dW[i - 1])

    return S_path


def delta_hedge(
        S_0,
        K,
        r,
        T,
        sigma,
        N,
        q=0.0,
        seed=None,
        option_type="call",
        position="short",
):
    
    if q != 0.0:
        raise NotImplementedError("Delta hedging currently supports q=0 only.")
    
    if position not in ("long", "short"):
        raise ValueError("position must be 'long' or 'short'")
    
    # Construct GBM path
    S_path = simulate_gbm_path(S_0, T, sigma, N, r, q, seed)
    dt = T / N
    
    # Calculate initial option price and delta
    price_0 = black_scholes_price(S=S_0, K=K, r=r, T=T, sigma=sigma, q=q, option_type=option_type)
    delta_0 = BS_delta(S=S_0, K=K, r=r, T=T, sigma=sigma, q=q, option_type=option_type)
    
    # Set up initial hedge
    position_sign = 1 if position == "long" else -1
    stock_position_0 = -position_sign * delta_0
    B_0 = -position_sign * price_0 - stock_position_0 * S_0
    
    # Store hedge values
    cash_account = [B_0]
    delta_path = [delta_0]
    stock_positions = [stock_position_0]
    option_prices = [price_0]
    
    # Rebalance before maturity
    for i in range(1, N):
        
        tau_i = T - i * dt
        
        price_i = black_scholes_price(S=S_path[i], K=K, r=r, T=tau_i, sigma=sigma, q=q, option_type=option_type)
        delta_i = BS_delta(S=S_path[i], K=K, r=r, T=tau_i, sigma=sigma, q=q, option_type=option_type)
        
        stock_position_i = -position_sign * delta_i
        cash_account_i = cash_account[i - 1] * np.exp(r * dt) - (stock_position_i - stock_positions[i - 1]) * S_path[i]
        
        option_prices.append(price_i)
        delta_path.append(delta_i)
        stock_positions.append(stock_position_i)
        cash_account.append(cash_account_i)


    # Close the hedge at maturity
    S_T = S_path[-1]
    cash_at_maturity = cash_account[-1] * np.exp(r * dt)
    stock_liquidation = stock_positions[-1] * S_T
    
    if option_type == "call":
        payoff = max(S_T - K, 0.0)
    else:
        payoff = max(K - S_T, 0.0)
    
    terminal_pnl = cash_at_maturity + stock_liquidation + position_sign * payoff
    
    
    return {
    "S_path": S_path,
    "option_prices": np.array(option_prices),
    "delta_path": np.array(delta_path),
    "stock_positions": np.array(stock_positions),
    "cash_account": np.array(cash_account),
    "payoff": payoff,
    "terminal_pnl": terminal_pnl,
    }

