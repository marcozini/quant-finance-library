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


# Load model settings from the Excel steering file.

def load_model_settings(file_path):
    """
    Load model settings from the Excel steering file.

    Parameters
    ----------
    file_path : str
        Path to the Excel input file.

    Returns
    -------
    dict
        Dictionary containing the model settings.
    """

    settings_data = pd.read_excel(
        file_path,
        sheet_name="model_settings",
    )

    if (
        "setting" not in settings_data.columns
        or "value" not in settings_data.columns
    ):
        raise ValueError(
            "The model_settings sheet must contain "
            "'setting' and 'value' columns."
        )

    settings = dict(
        zip(
            settings_data["setting"],
            settings_data["value"],
        )
    )

    return settings


# Load factor proxy mappings from the Excel steering file.

def load_factor_proxies(file_path):
    """
    Load factor proxy mappings from the Excel steering file.

    Parameters
    ----------
    file_path : str
        Path to the Excel input file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing factor proxy mappings.
    """

    factor_proxies = pd.read_excel(
        file_path,
        sheet_name="factor_proxies",
    )

    return factor_proxies


# Load rating migration matrix and credit spread inputs.

def load_rating_migration_matrix(file_path):
    """
    Load rating transition probabilities and credit spreads
    from the Excel steering file.

    Parameters
    ----------
    file_path : str
        Path to the Excel input file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing rating migration probabilities
        and credit spread inputs.
    """

    rating_migration_matrix = pd.read_excel(
        file_path,
        sheet_name="rating_migration_matrix",
    )

    return rating_migration_matrix


# Validate the basic structure and values of the credit portfolio.

def validate_portfolio(portfolio):

    required_columns = [
        "counterparty",
        "ticker",
        "factor_loading_proxy_ticker",
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

    # LGD must lie between zero and one.
    if (
        (portfolio["lgd"] < 0)
        | (portfolio["lgd"] > 1)
    ).any():
        raise ValueError(
            "LGD must be between 0 and 1."
        )

    # PD method must be either rating or Merton.
    valid_pd_methods = {
        "rating",
        "merton",
    }

    if not portfolio[
        "pd_method"
    ].isin(valid_pd_methods).all():
        raise ValueError(
            "pd_method must be either 'rating' or 'merton'."
        )

    # Rating-based counterparties must have a rating.
    rating_rows = (
        portfolio["pd_method"]
        == "rating"
    )

    if portfolio.loc[
        rating_rows,
        "rating",
    ].isna().any():
        raise ValueError(
            "A rating must be provided when pd_method is 'rating'."
        )

    # Merton counterparties need either a ticker
    # or complete manual Merton inputs.
    merton_rows = (
        portfolio["pd_method"]
        == "merton"
    )

    manual_merton_columns = [
        "equity_value",
        "equity_volatility",
        "debt",
        "risk_free_rate",
    ]

    for _, row in portfolio.loc[
        merton_rows
    ].iterrows():

        has_ticker = (
            pd.notna(row["ticker"])
            and str(
                row["ticker"]
            ).strip() != ""
        )

        has_manual_inputs = (
            row[
                manual_merton_columns
            ]
            .notna()
            .all()
        )

        if (
            not has_ticker
            and not has_manual_inputs
        ):
            raise ValueError(
                f"Merton inputs are incomplete for counterparty "
                f"'{row['counterparty']}'. Provide either a ticker "
                f"or complete manual Merton inputs."
            )

    # Unlisted counterparties need a listed proxy
    # for factor-loading estimation.
    for _, row in portfolio.iterrows():

        has_ticker = (
            pd.notna(row["ticker"])
            and str(
                row["ticker"]
            ).strip() != ""
        )

        has_factor_loading_proxy = (
            pd.notna(
                row[
                    "factor_loading_proxy_ticker"
                ]
            )
            and str(
                row[
                    "factor_loading_proxy_ticker"
                ]
            ).strip() != ""
        )

        if (
            not has_ticker
            and not has_factor_loading_proxy
        ):
            raise ValueError(
                f"A factor-loading proxy ticker must be provided "
                f"for unlisted counterparty "
                f"'{row['counterparty']}'."
            )

    return True


# Validate model settings.

def validate_model_settings(settings):

    required_settings = [
        "number_simulations",
        "seed",
        "confidence_level",
        "dependence_model",
        "t_degrees_of_freedom",
        "factor_structure",
        "factor_lookback_years",
    ]

    # Check that all required settings exist.
    missing_settings = [
        setting
        for setting in required_settings
        if setting not in settings
    ]

    if missing_settings:
        raise ValueError(
            f"Missing model settings: {missing_settings}"
        )

    # Number of simulations must be a positive integer.
    number_simulations = settings[
        "number_simulations"
    ]

    if (
        not float(
            number_simulations
        ).is_integer()
        or number_simulations <= 0
    ):
        raise ValueError(
            "number_simulations must be a positive integer."
        )

    # Seed must be a non-negative integer.
    seed = settings["seed"]

    if (
        not float(
            seed
        ).is_integer()
        or seed < 0
    ):
        raise ValueError(
            "seed must be a non-negative integer."
        )

    # Confidence level must lie strictly between zero and one.
    confidence_level = settings[
        "confidence_level"
    ]

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    # Validate dependence model.
    valid_dependence_models = {
        "independent",
        "gaussian_copula",
        "t_copula",
    }

    dependence_model = settings[
        "dependence_model"
    ]

    if dependence_model not in valid_dependence_models:
        raise ValueError(
            "dependence_model must be 'independent', "
            "'gaussian_copula', or 't_copula'."
        )

    # Validate t-copula degrees of freedom.
    if dependence_model == "t_copula":

        t_degrees_of_freedom = settings[
            "t_degrees_of_freedom"
        ]

        if t_degrees_of_freedom <= 2:
            raise ValueError(
                "t_degrees_of_freedom must be greater than two "
                "when dependence_model is 't_copula'."
            )

    # Validate factor structure.
    valid_factor_structures = {
        "single_factor",
        "global_sector",
        "global_region",
        "global_sector_region",
    }

    factor_structure = settings[
        "factor_structure"
    ]

    if factor_structure not in valid_factor_structures:
        raise ValueError(
            "factor_structure must be 'single_factor', "
            "'global_sector', 'global_region', or "
            "'global_sector_region'."
        )

    # Factor lookback period must be positive.
    factor_lookback_years = settings[
        "factor_lookback_years"
    ]

    if factor_lookback_years <= 0:
        raise ValueError(
            "factor_lookback_years must be greater than zero."
        )

    return True


# Validate factor proxy mappings.

def validate_factor_proxies(factor_proxies):

    required_columns = [
        "factor_type",
        "factor_name",
        "ticker",
        "description",
    ]

    # Check required columns.
    missing_columns = [
        column
        for column in required_columns
        if column not in factor_proxies.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing factor proxy columns: {missing_columns}"
        )

    # Validate factor types.
    valid_factor_types = {
        "global",
        "region",
        "sector",
    }

    if not factor_proxies[
        "factor_type"
    ].isin(valid_factor_types).all():
        raise ValueError(
            "factor_type must be 'global', 'region', or 'sector'."
        )

    # Factor names and tickers must be provided.
    if factor_proxies[
        "factor_name"
    ].isna().any():
        raise ValueError(
            "factor_name must not be missing."
        )

    if factor_proxies[
        "ticker"
    ].isna().any():
        raise ValueError(
            "ticker must not be missing."
        )

    # Each factor type/name combination must be unique.
    if factor_proxies.duplicated(
        subset=[
            "factor_type",
            "factor_name",
        ]
    ).any():
        raise ValueError(
            "Each factor_type and factor_name combination "
            "must be unique."
        )

    # Exactly one global factor should be defined.
    number_global_factors = (
        factor_proxies[
            "factor_type"
        ]
        .eq("global")
        .sum()
    )

    if number_global_factors != 1:
        raise ValueError(
            "Exactly one global factor must be defined."
        )

    return True


# Validate rating migration matrix and credit spreads.

def validate_rating_migration_matrix(
    rating_migration_matrix,
):

    # Synthetic one-year rating transition matrix and spreads.
    # Values are for demonstration purposes only and do not
    # represent data from a rating agency.

    rating_states = [
        "AAA",
        "AA",
        "A",
        "BBB",
        "BB",
        "B",
        "CCC",
        "D",
    ]

    required_columns = [
        "current_rating",
        "spread_bps",
        "spread_decimal",
        *rating_states,
        "row_sum",
    ]

    # Check required columns.
    missing_columns = [
        column
        for column in required_columns
        if column not in rating_migration_matrix.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing rating migration columns: "
            f"{missing_columns}"
        )

    # Normalize current ratings.
    current_ratings = (
        rating_migration_matrix[
            "current_rating"
        ]
        .astype(str)
        .str.upper()
    )

    # Each rating state must appear exactly once.
    if set(current_ratings) != set(
        rating_states
    ):
        raise ValueError(
            "The migration matrix must contain exactly "
            "AAA, AA, A, BBB, BB, B, CCC, and D."
        )

    if current_ratings.duplicated().any():
        raise ValueError(
            "Each current rating must appear exactly once."
        )

    # Transition probabilities must lie between zero and one.
    transition_matrix = (
        rating_migration_matrix[
            rating_states
        ]
    )

    if (
        (transition_matrix < 0)
        | (transition_matrix > 1)
    ).any().any():
        raise ValueError(
            "Transition probabilities must be between "
            "zero and one."
        )

    # Every transition row must sum to one.
    calculated_row_sums = (
        transition_matrix.sum(
            axis=1
        )
    )

    if not (
        calculated_row_sums
        .sub(1.0)
        .abs()
        .le(1e-8)
        .all()
    ):
        raise ValueError(
            "Each rating transition row must sum to one."
        )

    # Non-default ratings need credit spreads.
    non_default_rows = (
        current_ratings != "D"
    )

    if rating_migration_matrix.loc[
        non_default_rows,
        "spread_decimal",
    ].isna().any():
        raise ValueError(
            "Every non-default rating must have "
            "a credit spread."
        )

    # Credit spreads must be non-negative.
    if (
        rating_migration_matrix.loc[
            non_default_rows,
            "spread_decimal",
        ]
        < 0
    ).any():
        raise ValueError(
            "Credit spreads must be non-negative."
        )

    # Spread decimal must equal spread bps / 10000.
    spread_difference = (
        rating_migration_matrix.loc[
            non_default_rows,
            "spread_bps",
        ]
        / 10000
        - rating_migration_matrix.loc[
            non_default_rows,
            "spread_decimal",
        ]
    ).abs()

    if not spread_difference.le(
        1e-10
    ).all():
        raise ValueError(
            "spread_decimal must equal "
            "spread_bps / 10000."
        )

    # Lower credit quality should have wider spreads.
    ordered_spreads = (
        rating_migration_matrix
        .assign(
            current_rating=current_ratings
        )
        .set_index(
            "current_rating"
        )
        .loc[
            rating_states[:-1],
            "spread_decimal",
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not (
        ordered_spreads[1:]
        > ordered_spreads[:-1]
    ).all():
        raise ValueError(
            "Credit spreads must increase as "
            "credit quality deteriorates."
        )

    # Default must be an absorbing state.
    indexed_matrix = (
        rating_migration_matrix
        .assign(
            current_rating=current_ratings
        )
        .set_index(
            "current_rating"
        )
    )

    default_row = indexed_matrix.loc[
        "D",
        rating_states,
    ]

    if (
        default_row["D"] != 1
        or default_row.drop(
            "D"
        ).sum() != 0
    ):
        raise ValueError(
            "Default must be an absorbing state."
        )

    return True