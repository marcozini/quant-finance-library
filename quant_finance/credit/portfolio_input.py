#### Credit Portfolio Input ####

import pandas as pd


# Load credit portfolio data from the Excel steering file.

def load_portfolio(file_path):
    """
    Load the credit portfolio from an Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel input file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the credit portfolio inputs.
    """

    portfolio = pd.read_excel(
        file_path,
        sheet_name="credit_portfolio",
    )

    return portfolio


# Validate the basic structure and values of the credit portfolio.

def validate_portfolio(portfolio):

    required_columns = [
        "counterparty",
        "ticker",
        "exposure",
        "currency",
        "lgd",
        "pd_method",
        "rating",
        "equity_value",
        "equity_volatility",
        "debt",
        "risk_free_rate",
        "maturity",
        "coupon_rate",
        "payment_frequency",
        "sector",
        "region",
    ]

    # Check that all required columns exist.
    missing_columns = [
        column
        for column in required_columns
        if column not in portfolio.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Exposure must be strictly positive.
    if (portfolio["exposure"] <= 0).any():
        raise ValueError(
            "Exposure must be strictly positive."
        )

    # LGD must lie between 0 and 1.
    if ((portfolio["lgd"] < 0) | (portfolio["lgd"] > 1)).any():
        raise ValueError(
            "LGD must be between 0 and 1."
        )

    # PD method must be either rating or Merton.
    valid_pd_methods = {
        "rating",
        "merton",
    }

    if not portfolio["pd_method"].isin(valid_pd_methods).all():
        raise ValueError(
            "pd_method must be either 'rating' or 'merton'."
        )

    # Rating-based counterparties must have a rating.
    rating_rows = portfolio["pd_method"] == "rating"

    if portfolio.loc[rating_rows, "rating"].isna().any():
        raise ValueError(
            "A rating must be provided when pd_method is 'rating'."
        )

    # Merton counterparties need either a ticker
    # or complete manual Merton inputs.
    merton_rows = portfolio["pd_method"] == "merton"

    manual_merton_columns = [
        "equity_value",
        "equity_volatility",
        "debt",
        "risk_free_rate",
    ]

    for _, row in portfolio.loc[merton_rows].iterrows():

        has_ticker = (
            pd.notna(row["ticker"])
            and str(row["ticker"]).strip() != ""
        )

        has_manual_inputs = (
            row[manual_merton_columns]
            .notna()
            .all()
        )

        if not has_ticker and not has_manual_inputs:
            raise ValueError(
                f"Merton inputs are incomplete for counterparty "
                f"'{row['counterparty']}'. Provide either a ticker "
                f"or complete manual Merton inputs."
            )

    return True