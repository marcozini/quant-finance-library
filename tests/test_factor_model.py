#### Unit Tests for Credit Factor Model ####

import numpy as np
import pandas as pd
import pytest

from quant_finance.credit.factor_model import (
    estimate_factor_loadings,
)


# 1. Factor loadings are recovered from synthetic data.

def test_estimate_factor_loadings():

    rng = np.random.default_rng(0)

    n = 2000

    dates = pd.date_range(
        start="2020-01-01",
        periods=n,
        freq="D",
    )

    global_factor = rng.normal(size=n)
    region_factor = rng.normal(size=n)
    sector_factor = rng.normal(size=n)
    idiosyncratic = rng.normal(size=n)

    company_returns = (
        0.50 * global_factor
        + 0.30 * region_factor
        + 0.20 * sector_factor
        + np.sqrt(
            1
            - 0.50**2
            - 0.30**2
            - 0.20**2
        ) * idiosyncratic
    )

    result = estimate_factor_loadings(
        company_returns=pd.Series(
            company_returns,
            index=dates,
        ),
        global_returns=pd.Series(
            global_factor,
            index=dates,
        ),
        region_returns=pd.Series(
            region_factor,
            index=dates,
        ),
        sector_returns=pd.Series(
            sector_factor,
            index=dates,
        ),
    )

    assert result["global_loading"] == pytest.approx(
        0.50,
        abs=0.05,
    )

    assert result["region_loading"] == pytest.approx(
        0.30,
        abs=0.05,
    )

    assert result["sector_loading"] == pytest.approx(
        0.20,
        abs=0.05,
    )


# 2. Too few observations raise an error.

def test_factor_model_too_few_observations():

    dates = pd.date_range(
        start="2025-01-01",
        periods=20,
        freq="D",
    )

    returns = pd.Series(
        np.linspace(
            -0.01,
            0.01,
            20,
        ),
        index=dates,
    )

    with pytest.raises(ValueError):

        estimate_factor_loadings(
            company_returns=returns,
            global_returns=returns,
            region_returns=returns,
            sector_returns=returns,
        )