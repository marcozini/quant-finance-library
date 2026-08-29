#### Credit Factor Model ####

import numpy as np
import pandas as pd


# Standardize a return series to mean zero and unit variance.

def _standardize(values):

    standard_deviation = values.std(ddof=1)

    if standard_deviation <= 0:
        raise ValueError(
            "Return series must have positive volatility."
        )

    return (
        values - values.mean()
    ) / standard_deviation


# Remove variation explained by existing factors.

def _residualize(target, factors):

    design_matrix = np.column_stack(
        [
            np.ones(len(target)),
            factors,
        ]
    )

    coefficients = np.linalg.lstsq(
        design_matrix,
        target,
        rcond=None,
    )[0]

    fitted_values = (
        design_matrix
        @ coefficients
    )

    return target - fitted_values


# Estimate global, regional, and sector factor loadings.

def estimate_factor_loadings(
    company_returns,
    global_returns,
    region_returns,
    sector_returns,
):
    """
    Estimate orthogonal global, regional, and sector
    factor loadings from historical equity returns.

    The factor structure is constructed sequentially:

    1. Global factor.
    2. Regional factor net of the global factor.
    3. Sector factor net of global and regional factors.

    Parameters
    ----------
    company_returns : pandas.Series
        Counterparty equity returns.

    global_returns : pandas.Series
        Global market proxy returns.

    region_returns : pandas.Series
        Regional market proxy returns.

    sector_returns : pandas.Series
        Sector market proxy returns.

    Returns
    -------
    dict
        Global, regional, sector, and idiosyncratic loadings.
    """

    # Align all return series by date.
    returns = pd.concat(
        [
            company_returns.rename("company"),
            global_returns.rename("global"),
            region_returns.rename("region"),
            sector_returns.rename("sector"),
        ],
        axis=1,
    ).dropna()

    if len(returns) < 60:
        raise ValueError(
            "At least 60 aligned return observations are required."
        )

    # Standardize company and factor returns.
    company = _standardize(
        returns["company"]
    )

    global_factor = _standardize(
        returns["global"]
    )

    region_raw = _standardize(
        returns["region"]
    )

    sector_raw = _standardize(
        returns["sector"]
    )


    # Remove the global component from the regional factor.
    region_residual = _residualize(
        target=region_raw.to_numpy(),
        factors=global_factor.to_numpy(),
    )

    region_factor = _standardize(
        pd.Series(
            region_residual,
            index=returns.index,
        )
    )


    # Remove global and regional components from the sector factor.
    existing_factors = np.column_stack(
        [
            global_factor.to_numpy(),
            region_factor.to_numpy(),
        ]
    )

    sector_residual = _residualize(
        target=sector_raw.to_numpy(),
        factors=existing_factors,
    )

    sector_factor = _standardize(
        pd.Series(
            sector_residual,
            index=returns.index,
        )
    )


    # Estimate company loadings on the orthogonal factors.
    factor_matrix = np.column_stack(
        [
            global_factor.to_numpy(),
            region_factor.to_numpy(),
            sector_factor.to_numpy(),
        ]
    )

    loadings = np.linalg.lstsq(
        factor_matrix,
        company.to_numpy(),
        rcond=None,
    )[0]

    global_loading = float(
        loadings[0]
    )

    region_loading = float(
        loadings[1]
    )

    sector_loading = float(
        loadings[2]
    )


    # Remaining variance is attributed to idiosyncratic risk.
    systematic_variance = (
        global_loading**2
        + region_loading**2
        + sector_loading**2
    )

    if systematic_variance > 1 + 1e-10:
        raise ValueError(
            "Estimated systematic variance exceeds one."
        )

    idiosyncratic_loading = np.sqrt(
        max(
            0.0,
            1.0 - systematic_variance,
        )
    )


    return {
        "global_loading": global_loading,
        "region_loading": region_loading,
        "sector_loading": sector_loading,
        "idiosyncratic_loading": float(
            idiosyncratic_loading
        ),
    }