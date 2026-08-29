#### Credit Factor Market Data ####

import numpy as np
import yfinance as yf

from quant_finance.credit.factor_model import (
    estimate_factor_loadings,
)


# Download historical daily log returns for an equity or ETF.

def _download_log_returns(
    ticker,
    start_date,
    end_date,
):
    """
    Download historical adjusted prices and calculate
    daily log returns.
    """

    market_data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if market_data.empty:
        raise ValueError(
            f"No market data found for ticker '{ticker}'."
        )

    adjusted_prices = (
        market_data["Adj Close"]
        .squeeze()
        .dropna()
    )

    if len(adjusted_prices) < 2:
        raise ValueError(
            f"Not enough price data for ticker '{ticker}'."
        )

    log_returns = np.log(
        adjusted_prices
        / adjusted_prices.shift(1)
    ).dropna()

    return log_returns


# Find the market proxy ticker for a factor.

def get_factor_proxy_ticker(
    factor_proxies,
    factor_type,
    factor_name=None,
):
    """
    Retrieve the configured market proxy ticker
    for a systematic factor.
    """

    matching_rows = factor_proxies[
        factor_proxies["factor_type"]
        == factor_type
    ]

    if factor_name is not None:

        matching_rows = matching_rows[
            matching_rows["factor_name"]
            == factor_name
        ]

    if len(matching_rows) != 1:
        raise ValueError(
            f"Exactly one factor proxy must exist for "
            f"factor_type='{factor_type}' "
            f"and factor_name='{factor_name}'."
        )

    ticker = str(
        matching_rows.iloc[0]["ticker"]
    ).strip()

    return ticker


# Estimate factor loadings from historical market data.

def estimate_market_factor_loadings(
    company_ticker,
    global_ticker,
    region_ticker,
    sector_ticker,
    start_date,
    end_date,
):
    """
    Estimate global, regional, sector, and idiosyncratic
    loadings from historical market data.
    """

    company_returns = _download_log_returns(
        ticker=company_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    global_returns = _download_log_returns(
        ticker=global_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    region_returns = _download_log_returns(
        ticker=region_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    sector_returns = _download_log_returns(
        ticker=sector_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    factor_loadings = estimate_factor_loadings(
        company_returns=company_returns,
        global_returns=global_returns,
        region_returns=region_returns,
        sector_returns=sector_returns,
    )

    return factor_loadings


# Estimate counterparty factor loadings using configured proxies.

def estimate_counterparty_factor_loadings(
    company_ticker,
    region,
    sector,
    factor_proxies,
    start_date,
    end_date,
):
    """
    Estimate counterparty factor loadings using the global,
    regional, and sector proxies defined in the Excel input.
    """

    global_ticker = get_factor_proxy_ticker(
        factor_proxies=factor_proxies,
        factor_type="global",
    )

    region_ticker = get_factor_proxy_ticker(
        factor_proxies=factor_proxies,
        factor_type="region",
        factor_name=region,
    )

    sector_ticker = get_factor_proxy_ticker(
        factor_proxies=factor_proxies,
        factor_type="sector",
        factor_name=sector,
    )

    factor_loadings = estimate_market_factor_loadings(
        company_ticker=company_ticker,
        global_ticker=global_ticker,
        region_ticker=region_ticker,
        sector_ticker=sector_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    return factor_loadings