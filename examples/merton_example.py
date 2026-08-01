#### Merton Model ####

## Synthetic example

import numpy as np
from scipy.stats import norm

from quant_finance.credit.merton import (
    merton_equity_value,
    merton_debt_value,
    distance_to_default,
    merton_probability_of_default,
)
from quant_finance.credit.merton_calibration import calibrate_merton


#Assume true, normally unobservable firm parameters
true_asset_value = 150.0
true_asset_volatility = 0.20
debt_face_value = 100.0
risk_free_rate = 0.03
time_to_maturity = 1.0


#Calculate d1 from the true firm parameters
d1 = (np.log(true_asset_value / debt_face_value) +
      (risk_free_rate + 0.5 * true_asset_volatility**2)
      * time_to_maturity) / (
      true_asset_volatility * np.sqrt(time_to_maturity))


#Generate synthetic observable equity data
equity_value = merton_equity_value(
    asset_value=true_asset_value,
    debt_face_value=debt_face_value,
    risk_free_rate=risk_free_rate,
    asset_volatility=true_asset_volatility,
    time_to_maturity=time_to_maturity,
)

equity_volatility = (
    true_asset_value / equity_value
    * norm.cdf(d1)
    * true_asset_volatility
)


#Recover the hidden asset parameters through calibration
calibrated_asset_value, calibrated_asset_volatility = calibrate_merton(
    equity_value=equity_value,
    equity_volatility=equity_volatility,
    debt=debt_face_value,
    risk_free_rate=risk_free_rate,
    maturity=time_to_maturity,
)


#Compare calibrated and true parameters
print("Synthetic observable equity data")
print(f"Equity value:      {equity_value:.6f}")
print(f"Equity volatility: {equity_volatility:.6f}")

print("\nAsset value")
print(f"True:       {true_asset_value:.6f}")
print(f"Calibrated: {calibrated_asset_value:.6f}")

print("\nAsset volatility")
print(f"True:       {true_asset_volatility:.6f}")
print(f"Calibrated: {calibrated_asset_volatility:.6f}")


#Use the calibrated parameters in the core Merton model
debt_value = merton_debt_value(
    asset_value=calibrated_asset_value,
    debt_face_value=debt_face_value,
    risk_free_rate=risk_free_rate,
    asset_volatility=calibrated_asset_volatility,
    time_to_maturity=time_to_maturity,
)

dd = distance_to_default(
    asset_value=calibrated_asset_value,
    debt_face_value=debt_face_value,
    risk_free_rate=risk_free_rate,
    asset_volatility=calibrated_asset_volatility,
    time_to_maturity=time_to_maturity,
)

pd = merton_probability_of_default(
    asset_value=calibrated_asset_value,
    debt_face_value=debt_face_value,
    risk_free_rate=risk_free_rate,
    asset_volatility=calibrated_asset_volatility,
    time_to_maturity=time_to_maturity,
)


#Display the resulting credit-risk measures
print("\nMerton credit-risk results")
print(f"Risky debt value:       {debt_value:.6f}")
print(f"Distance to default:    {dd:.6f}")
print(f"Probability of default: {pd:.6%}")