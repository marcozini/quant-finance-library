#### Unit tests Merton ####

import numpy as np
import pytest
from scipy.stats import norm

from quant_finance.credit.merton import merton_equity_value
from quant_finance.credit.merton_calibration import (
    calibrate_merton,
    merton_calibration_residuals,
)


# Consistent synthetic inputs
true_asset_value = 150.0
true_asset_volatility = 0.15
debt = 100.0
risk_free_rate = 0.03
maturity = 1.0

equity_value = merton_equity_value(
    true_asset_value,
    debt,
    risk_free_rate,
    true_asset_volatility,
    maturity,
)

d1 = (
    np.log(true_asset_value / debt)
    + (risk_free_rate + 0.5 * true_asset_volatility**2) * maturity
) / (true_asset_volatility * np.sqrt(maturity))

equity_volatility = (
    true_asset_value / equity_value
    * norm.cdf(d1)
    * true_asset_volatility
)


# 1. True asset parameters produce approximately zero calibration residuals
def test_residuals_are_zero_for_true_parameters():
    
    residuals = merton_calibration_residuals(
        [true_asset_value, true_asset_volatility],
        equity_value,
        equity_volatility,
        debt,
        risk_free_rate,
        maturity,
    )

    assert residuals == pytest.approx([0.0, 0.0], abs=1e-10)


# 2. Calibration recovers the known asset value and asset volatility
def test_calibration_recovers_true_parameters():
    
    asset_value, asset_volatility = calibrate_merton(
        equity_value,
        equity_volatility,
        debt,
        risk_free_rate,
        maturity,
    )

    assert asset_value == pytest.approx(true_asset_value, rel=1e-8)
    assert asset_volatility == pytest.approx(
        true_asset_volatility,
        rel=1e-8,
    )