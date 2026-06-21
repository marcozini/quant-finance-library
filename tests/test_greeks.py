#### Unit tests for Black-Scholes Greeks ####

import pytest

from quant_finance.options.greeks import BS_delta, BS_gamma, BS_vega


## Benchmark Greeks for:
# S = 100, K = 100, r = 1%, T = 1, sigma = 20%, q = 0


# a) Call delta
def test_black_scholes_call_delta_known_value():
    value = BS_delta(
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        sigma = 0.2,
        q = 0.0,
        option_type = "call",
    )

    assert value == pytest.approx(0.5596, rel = 1e-4)


# b) Put delta
def test_black_scholes_put_delta_known_value():
    value = BS_delta(
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        sigma = 0.2,
        q = 0.0,
        option_type = "put",
    )

    assert value == pytest.approx(-0.4404, rel = 1e-4)


# c) Gamma
def test_black_scholes_gamma_known_value():
    value = BS_gamma(
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        sigma = 0.2,
        q = 0.0,
    )

    assert value == pytest.approx(0.019724, rel = 1e-4)


# d) Vega
def test_black_scholes_vega_known_value():
    value = BS_vega(
        S = 100,
        K = 100,
        r = 0.01,
        T = 1,
        sigma = 0.2,
        q = 0.0,
    )

    assert value == pytest.approx(39.4479, rel = 1e-4)