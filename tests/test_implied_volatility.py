#### Unit tests for implied volatility ####

import pytest

from quant_finance.options.black_scholes_pricing import black_scholes_price
from quant_finance.options.implied_volatility import implied_volatility


## Implied volatility recovery test:
# Generate a Black-Scholes price with sigma = 20%.
# Then check whether the solver recovers sigma = 20%.


# a) Bisection method
def test_implied_volatility_bisection_recovers_true_sigma():
    sigma_true = 0.2

    market_price = black_scholes_price(
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        sigma = sigma_true,
        q = 0.0,
        option_type = "call",
    )

    iv = implied_volatility(
        C = market_price,
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        q = 0.0,
        option_type = "call",
        method = "bisection",
    )

    assert iv == pytest.approx(sigma_true, rel = 1e-4)


# b) Newton-Raphson method
def test_implied_volatility_newton_recovers_true_sigma():
    sigma_true = 0.2

    market_price = black_scholes_price(
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        sigma = sigma_true,
        q = 0.0,
        option_type = "call",
    )

    iv = implied_volatility(
        C = market_price,
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        q = 0.0,
        option_type = "call",
        method = "newton",
    )

    assert iv == pytest.approx(sigma_true, rel = 1e-4)


# c) Invalid method
def test_implied_volatility_invalid_method_raises_error():
    with pytest.raises(ValueError):
        implied_volatility(
            C = 8.4333,
            S = 100,
            K = 100,
            r = 0.01,
            T = 1,
            q = 0.0,
            option_type = "call",
            method = "invalid",
        )