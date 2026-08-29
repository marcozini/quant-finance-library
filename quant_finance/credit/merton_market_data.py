#### Merton Market Data ####

import numpy as np
import pandas as pd
import yfinance as yf


# Estimate annualized equity volatility from historical market data.

def estimate_equity_volatility(
    ticker,
    start_date,
    end_date,
):
    """
    Estimate annualized equity volatility from historical daily returns.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol.

    start_date : str
        Start date for the historical price data.

    end_date : str
        End date for the historical price data.

    Returns
    -------
    float
        Annualized equity volatility.
    """

    # Download historical equity prices.
    market_data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    # Check that market data was returned.
    if market_data.empty:
        raise ValueError(
            f"No market data found for ticker '{ticker}'."
        )

    # Use adjusted closing prices for return calculation.
    adjusted_prices = (
        market_data["Adj Close"]
        .squeeze()
        .dropna()
    )

    # Calculate daily log returns.
    log_returns = np.log(
        adjusted_prices / adjusted_prices.shift(1)
    ).dropna()

    # Check that enough observations are available.
    if len(log_returns) < 2:
        raise ValueError(
            f"Not enough price data to estimate volatility "
            f"for ticker '{ticker}'."
        )

    # Annualize daily equity volatility.
    equity_volatility = (
        log_returns.std(ddof=1)
        * np.sqrt(252)
    )

    return equity_volatility


# Estimate the market value of equity.

def estimate_equity_value(ticker):
    """
    Estimate the market value of equity from the latest available
    share price and shares outstanding.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol.

    Returns
    -------
    float
        Estimated market value of equity.
    """

    stock = yf.Ticker(ticker)

    # Retrieve company information.
    company_info = stock.get_info()

    shares_outstanding = company_info.get(
        "sharesOutstanding"
    )

    if shares_outstanding is None:
        raise ValueError(
            f"No shares outstanding found for ticker '{ticker}'."
        )

    if shares_outstanding <= 0:
        raise ValueError(
            f"Shares outstanding must be positive for ticker '{ticker}'."
        )

    # Retrieve recent prices and use the latest available close.
    price_data = stock.history(
        period="5d",
        auto_adjust=False,
    )

    if price_data.empty:
        raise ValueError(
            f"No recent price data found for ticker '{ticker}'."
        )

    closing_prices = price_data["Close"].dropna()

    if closing_prices.empty:
        raise ValueError(
            f"No valid closing price found for ticker '{ticker}'."
        )

    latest_price = closing_prices.iloc[-1]

    # Equity market value = share price × shares outstanding.
    equity_value = (
        latest_price
        * shares_outstanding
    )

    return float(equity_value)


# Estimate the Merton debt default point from balance-sheet data.

def estimate_debt_default_point(ticker):
    """
    Estimate the Merton debt default point using the latest
    available annual balance-sheet data.

    The default point is approximated as:

        short-term debt + 0.5 * long-term debt

    Parameters
    ----------
    ticker : str
        Equity ticker symbol.

    Returns
    -------
    tuple
        Estimated default point, short-term debt, and long-term debt.
    """

    stock = yf.Ticker(ticker)

    # Retrieve annual balance-sheet data.
    balance_sheet = stock.get_balance_sheet(
        freq="yearly"
    )

    if balance_sheet.empty:
        raise ValueError(
            f"No balance-sheet data found for ticker '{ticker}'."
        )

    # Use the latest available reporting date.
    latest_date = max(balance_sheet.columns)

    # Possible Yahoo Finance labels for short-term debt.
    short_term_candidates = [
        "CurrentDebt",
        "CurrentDebtAndCapitalLeaseObligation",
    ]

    # Possible Yahoo Finance labels for long-term debt.
    long_term_candidates = [
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligation",
    ]

    short_term_debt = None
    long_term_debt = None

    # Find short-term debt.
    for item in short_term_candidates:

        if item in balance_sheet.index:

            value = balance_sheet.loc[
                item,
                latest_date,
            ]

            if pd.notna(value):
                short_term_debt = float(value)
                break

    # Find long-term debt.
    for item in long_term_candidates:

        if item in balance_sheet.index:

            value = balance_sheet.loc[
                item,
                latest_date,
            ]

            if pd.notna(value):
                long_term_debt = float(value)
                break

    if short_term_debt is None:
        raise ValueError(
            f"No short-term debt found for ticker '{ticker}'."
        )

    if long_term_debt is None:
        raise ValueError(
            f"No long-term debt found for ticker '{ticker}'."
        )

    # KMV-style approximation of the default point.
    debt_default_point = (
        short_term_debt
        + 0.5 * long_term_debt
    )

    return (
        debt_default_point,
        short_term_debt,
        long_term_debt,
    )