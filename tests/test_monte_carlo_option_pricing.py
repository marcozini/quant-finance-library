#### Unit tests for Monte Carlo option pricing ####

import pytest

from quant_finance.options.monte_carlo_option_pricing import monte_carlo_option_price
from quant_finance.options.black_scholes_pricing import black_scholes_price



# Call option
def test_monte_carlo_call_option_known_value():
    
    bs_price = black_scholes_price(
        S=100,
        K=100,
        r=0.05,
        T=1,
        sigma=0.20,
        q=0.0,
        option_type='call')
    
    mc_price = monte_carlo_option_price(
        K=100,
        S_0=100,
        r=0.05,
        sigma=0.20,
        T=1,
        q=0.0,
        nsim=100000,
        option_type='call',
        alpha=0.05,
        return_stats=False,
        seed=0
        )
    
    assert mc_price == pytest.approx(bs_price, rel = 1e-2)
    
    
# Put option
def test_monte_carlo_put_option_known_value():
    
    bs_price = black_scholes_price(
        S=100,
        K=100,
        r=0.05,
        T=1,
        sigma=0.20,
        q=0.0,
        option_type="put")
    
    mc_price = monte_carlo_option_price(
        K=100,
        S_0=100,
        r=0.05,
        sigma=0.20,
        T=1,
        q=0.0,
        nsim=100000,
        option_type='put',
        alpha=0.05,
        return_stats=False,
        seed=0
        )
    
    assert mc_price == pytest.approx(bs_price, rel = 1e-2)
    
    
# Statistis (i.e standard error and confidence intervals)
def test_monte_carlo_return_stats():

    mc_price, standard_error, confidence_interval = monte_carlo_option_price(
        K=100,
        S_0=100,
        r=0.05,
        sigma=0.20,
        T=1,
        q=0.0,
        nsim=100000,
        option_type="call",
        alpha=0.05,
        return_stats=True,
        seed=0,
    )

    ci_lower, ci_upper = confidence_interval

    assert standard_error > 0
    assert ci_lower < mc_price < ci_upper
    
    
# Raise Error Test
def test_invalid_option_type_raises_error():

    with pytest.raises(ValueError):
        monte_carlo_option_price(
            K=100,
            S_0=100,
            r=0.05,
            sigma=0.20,
            T=1,
            q=0.0,
            nsim=1000,
            option_type="invalid",
        )
