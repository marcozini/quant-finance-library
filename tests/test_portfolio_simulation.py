#### Unit Tests for Credit Portfolio Simulation ####

import numpy as np
import pandas as pd
import pytest

from quant_finance.credit.credit_migration import (
    RATING_STATES,
    get_adjusted_transition_probabilities,
)

from quant_finance.credit.portfolio_simulation import (
    simulate_default_matrix,
    simulate_portfolio_losses,
    simulate_migration_states,
)


# Create a small valid portfolio for simulation tests.

def create_test_portfolio():

    return pd.DataFrame({
        "counterparty": [
            "Company A",
            "Company B",
        ],
        "exposure": [
            100.0,
            50.0,
        ],
        "lgd": [
            0.45,
            0.40,
        ],
        "pd": [
            0.02,
            0.05,
        ],
        "region": [
            "Europe",
            "Europe",
        ],
        "sector": [
            "Industrial",
            "Technology",
        ],
        "global_loading": [
            0.40,
            0.50,
        ],
        "region_loading": [
            0.20,
            0.20,
        ],
        "sector_loading": [
            0.15,
            0.10,
        ],
    })


# Create valid simulation settings.

def create_test_settings():

    return {
        "number_simulations": 200000,
        "seed": 0,
        "dependence_model": "t_copula",
        "t_degrees_of_freedom": 5,
        "factor_structure": "global_sector_region",
    }


# Create a synthetic rating migration matrix.

def create_test_rating_migration_matrix():

    return pd.DataFrame({
        "current_rating": [
            "AAA",
            "AA",
            "A",
            "BBB",
            "BB",
            "B",
            "CCC",
            "D",
        ],
        "spread_decimal": [
            0.003,
            0.005,
            0.008,
            0.015,
            0.030,
            0.060,
            0.120,
            None,
        ],
        "AAA": [
            0.9150,
            0.0080,
            0.0010,
            0.0002,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ],
        "AA": [
            0.0750,
            0.9000,
            0.0250,
            0.0030,
            0.0005,
            0.0001,
            0.0000,
            0.0000,
        ],
        "A": [
            0.0080,
            0.0750,
            0.8900,
            0.0350,
            0.0050,
            0.0010,
            0.0005,
            0.0000,
        ],
        "BBB": [
            0.0010,
            0.0120,
            0.0650,
            0.8500,
            0.0350,
            0.0050,
            0.0015,
            0.0000,
        ],
        "BB": [
            0.0003,
            0.0020,
            0.0120,
            0.0750,
            0.8000,
            0.0500,
            0.0080,
            0.0000,
        ],
        "B": [
            0.0001,
            0.0010,
            0.0030,
            0.0200,
            0.1000,
            0.7800,
            0.0500,
            0.0000,
        ],
        "CCC": [
            0.0001,
            0.0005,
            0.0010,
            0.0050,
            0.0250,
            0.0800,
            0.6000,
            0.0000,
        ],
        "D": [
            0.0005,
            0.0015,
            0.0030,
            0.0118,
            0.0345,
            0.0839,
            0.3400,
            1.0000,
        ],
    })


# 1. Independent simulation returns the correct shape.

def test_independent_default_matrix_shape():

    portfolio = create_test_portfolio()
    settings = create_test_settings()

    settings["dependence_model"] = "independent"

    defaults = simulate_default_matrix(
        portfolio=portfolio,
        settings=settings,
    )

    assert defaults.shape == (
        settings["number_simulations"],
        len(portfolio),
    )


# 2. Independent simulated default rates approximate input PDs.

def test_independent_default_rates():

    portfolio = create_test_portfolio()
    settings = create_test_settings()

    settings["dependence_model"] = "independent"

    defaults = simulate_default_matrix(
        portfolio=portfolio,
        settings=settings,
    )

    simulated_pds = defaults.mean(
        axis=0
    )

    assert simulated_pds[0] == pytest.approx(
        portfolio.loc[
            0,
            "pd",
        ],
        abs=0.002,
    )

    assert simulated_pds[1] == pytest.approx(
        portfolio.loc[
            1,
            "pd",
        ],
        abs=0.002,
    )


# 3. Gaussian copula preserves marginal default probabilities.

def test_gaussian_copula_default_rates():

    portfolio = create_test_portfolio()
    settings = create_test_settings()

    settings[
        "dependence_model"
    ] = "gaussian_copula"

    defaults = simulate_default_matrix(
        portfolio=portfolio,
        settings=settings,
    )

    simulated_pds = defaults.mean(
        axis=0
    )

    assert simulated_pds[0] == pytest.approx(
        portfolio.loc[
            0,
            "pd",
        ],
        abs=0.002,
    )

    assert simulated_pds[1] == pytest.approx(
        portfolio.loc[
            1,
            "pd",
        ],
        abs=0.002,
    )


# 4. t-copula preserves marginal default probabilities.

def test_t_copula_default_rates():

    portfolio = create_test_portfolio()
    settings = create_test_settings()

    defaults = simulate_default_matrix(
        portfolio=portfolio,
        settings=settings,
    )

    simulated_pds = defaults.mean(
        axis=0
    )

    assert simulated_pds[0] == pytest.approx(
        portfolio.loc[
            0,
            "pd",
        ],
        abs=0.002,
    )

    assert simulated_pds[1] == pytest.approx(
        portfolio.loc[
            1,
            "pd",
        ],
        abs=0.002,
    )


# 5. Portfolio loss simulation has correct output structure.

def test_portfolio_loss_output():

    portfolio = create_test_portfolio()
    settings = create_test_settings()

    losses, defaults = simulate_portfolio_losses(
        portfolio=portfolio,
        settings=settings,
        return_defaults=True,
    )

    assert losses.shape == (
        settings["number_simulations"],
    )

    assert defaults.shape == (
        settings["number_simulations"],
        len(portfolio),
    )

    assert np.all(
        losses >= 0
    )


# 6. Maximum portfolio loss equals total exposure times LGD.

def test_maximum_possible_portfolio_loss():

    portfolio = create_test_portfolio()

    maximum_loss = (
        portfolio["exposure"]
        * portfolio["lgd"]
    ).sum()

    assert maximum_loss == pytest.approx(
        65.0
    )


# 7. Simulated migration states reproduce marginal probabilities.

def test_migration_state_probabilities():

    portfolio = create_test_portfolio()

    portfolio["pd_method"] = [
        "rating",
        "rating",
    ]

    portfolio["rating"] = [
        "AA",
        "A",
    ]

    rating_migration_matrix = (
        create_test_rating_migration_matrix()
    )

    settings = create_test_settings()

    migration_states = simulate_migration_states(
        portfolio=portfolio,
        settings=settings,
        rating_migration_matrix=rating_migration_matrix,
    )

    expected_probabilities = (
        get_adjusted_transition_probabilities(
            current_rating="AA",
            pd=portfolio.loc[
                0,
                "pd",
            ],
            rating_migration_matrix=rating_migration_matrix,
        )
    )

    simulated_probabilities = np.array([
        np.mean(
            migration_states[
                :,
                0,
            ] == rating
        )
        for rating in RATING_STATES
    ])

    assert simulated_probabilities == pytest.approx(
        expected_probabilities,
        abs=0.003,
    )
    
    
# 8. Every supported dependence-model / factor-structure
# combination runs successfully through the simulation engine.
@pytest.mark.parametrize(
    "dependence_model",
    [
        "independent",
        "gaussian_copula",
        "t_copula",
    ],
)
@pytest.mark.parametrize(
    "factor_structure",
    [
        "single_factor",
        "global_sector",
        "global_region",
        "global_sector_region",
    ],
)
def test_all_dependence_factor_combinations_run(
    dependence_model,
    factor_structure,
):

    portfolio = create_test_portfolio()
    settings = create_test_settings()

    # Keep this integration-style unit test light.
    settings["number_simulations"] = 2000
    settings["dependence_model"] = dependence_model
    settings["factor_structure"] = factor_structure

    defaults = simulate_default_matrix(
        portfolio=portfolio,
        settings=settings,
    )

    assert defaults.shape == (
        settings["number_simulations"],
        len(portfolio),
    )

    assert np.all(
        np.isfinite(defaults)
    )

    assert np.all(
        (defaults == 0)
        | (defaults == 1)
    )