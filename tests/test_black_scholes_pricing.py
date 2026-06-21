####  Unit tests for Black-Scholes pricing ####

import pytest
import numpy as np

from quant_finance.options.black_scholes_pricing import black_scholes_price


##  Benchmark Black-Scholes price for:
# S = 100, K = 100, r = 1%, T=1, sigma = 20%, q = 0


# a) Call price
def test_black_scholes_call_price_known_value():
    price = black_scholes_price(
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        sigma = 0.2,
        q = 0.0,
        option_type = "call",
    )

    # pytest.approx allows small numerical tolerance.
    assert price == pytest.approx(8.4333, rel = 1e-4)
    
    
# b) put price
def test_black_scholes_put_price_known_value():
    price = black_scholes_price(
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        sigma = 0.2,
        q = 0.0,
        option_type = "put",
    )

    assert price == pytest.approx(7.4383, rel = 1e-4)
    
# c) Put-Call Parity
def test_black_scholes_put_call_parity():
    S = 100
    K = 100
    r = 0.01
    T = 1
    sigma = 0.2
    q = 0.0

    call_price = black_scholes_price(S, K, r, T, sigma, q, option_type = "call")
    put_price = black_scholes_price(S, K, r, T, sigma, q, option_type = "put")

    parity_left = call_price - put_price
    parity_right = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert parity_left == pytest.approx(parity_right, rel = 1e-4)
    
    
# d) Invalid option type
def test_black_scholes_invalid_option_type_raises_error():
    with pytest.raises(ValueError):
        black_scholes_price(
            S = 100,
            K = 100,
            r = 0.01,
            T = 1,
            sigma = 0.2,
            q = 0.0,
            option_type = "invalid", #Raise error if option type is not call or put
        )
    
