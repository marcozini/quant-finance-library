#### Unit Tests for binomial tree option pricing ####

import pytest

from quant_finance.options.black_scholes_pricing import black_scholes_price
from quant_finance.options.binomial_tree import (
    EuropeanOption,
    AmericanOption,
    BermudanOption,
)


# 1. European call price approximates Black-Scholes
def test_european_call_matches_black_scholes():
    binomial_option = EuropeanOption(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        N=500,
        q=0.0,
        option_type="call",
    )

    binomial_price, _ = binomial_option.price()

    black_scholes_value = black_scholes_price(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        q=0.0,
        option_type="call",
    )

    assert binomial_price == pytest.approx(
        black_scholes_value,
        abs=0.01,
    )


# 2. American call equals European call when q=0
def test_american_call_equals_european_call_without_dividends():
    european_call = EuropeanOption(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        N=500,
        q=0.0,
        option_type="call",
    )

    american_call = AmericanOption(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        N=500,
        q=0.0,
        option_type="call",
    )

    european_price, _ = european_call.price()
    american_price, _ = american_call.price()

    assert american_price == pytest.approx(
        european_price,
        abs=1e-10,
    )


# 3. For put options: European <= Bermudan <= American
def test_put_value_increases_with_exercise_flexibility():
    european_put = EuropeanOption(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        N=500,
        q=0.0,
        option_type="put",
    )

    bermudan_put = BermudanOption(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        N=500,
        q=0.0,
        option_type="put",
        exercise_steps={100, 200, 300, 400},
    )

    american_put = AmericanOption(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        N=500,
        q=0.0,
        option_type="put",
    )

    european_price, _ = european_put.price()
    bermudan_price, _ = bermudan_put.price()
    american_price, _ = american_put.price()

    assert european_price <= bermudan_price <= american_price


# 4. Stock and option trees have the correct shape
def test_binomial_tree_shape():
    option = EuropeanOption(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        N=3,
        q=0.0,
        option_type="call",
    )

    stock_tree = option.build_tree()
    _, option_tree = option.price()

    assert len(stock_tree) == 4
    assert len(option_tree) == 4

    for step in range(4):
        assert len(stock_tree[step]) == step + 1
        assert len(option_tree[step]) == step + 1


# 5. Invalid general inputs raise an error
def test_invalid_inputs_raise_error():
    with pytest.raises(ValueError):
        EuropeanOption(
            S0=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            N=0,
            q=0.0,
            option_type="call",
        )

    with pytest.raises(ValueError):
        EuropeanOption(
            S0=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            N=10,
            q=0.0,
            option_type="invalid",
        )


# 6. Invalid Bermudan exercise steps raise an error
def test_invalid_bermudan_exercise_steps_raise_error():
    with pytest.raises(ValueError):
        BermudanOption(
            S0=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            N=10,
            q=0.0,
            option_type="put",
            exercise_steps={2, 5, 11},
        )