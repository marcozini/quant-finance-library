#### Credit Portfolio Input ####

from datetime import date, datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Loading functions
# ---------------------------------------------------------

# Load credit portfolio data from the Excel steering file.
def load_portfolio(
    file_path,
):

    return pd.read_excel(
        file_path,
        sheet_name="credit_portfolio",
    )


# Load model settings from the Excel steering file.
def load_model_settings(
    file_path,
):

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

    return dict(
        zip(
            settings_data["setting"],
            settings_data["value"],
        )
    )


# Load factor proxy mappings from the Excel steering file.
def load_factor_proxies(
    file_path,
):

    return pd.read_excel(
        file_path,
        sheet_name="factor_proxies",
    )


# Load rating migration matrix and credit spread inputs.
def load_rating_migration_matrix(
    file_path,
):

    return pd.read_excel(
        file_path,
        sheet_name="rating_migration_matrix",
    )


# Load fallback market data snapshot.
def load_market_data(
    file_path,
):

    return pd.read_excel(
        file_path,
        sheet_name="market_data",
    )


# ---------------------------------------------------------
# Market data date
# ---------------------------------------------------------

# Parse the market data date from the Excel steering workbook.
#
# Supported inputs:
# - "latest"
# - YYYY-MM-DD
# - datetime.date
# - datetime.datetime
# - pandas.Timestamp
#
# "latest" is represented internally by None.
def parse_market_data_date(
    value,
):

    if (
        value is None
        or pd.isna(
            value
        )
    ):
        raise ValueError(
            "market_data_date must be 'latest' "
            "or a date in YYYY-MM-DD format."
        )

    # String input.
    if isinstance(
        value,
        str,
    ):

        value = (
            value
            .strip()
        )

        if (
            value.lower()
            == "latest"
        ):
            return None

        try:

            market_data_date = (
                datetime.strptime(
                    value,
                    "%Y-%m-%d",
                )
                .date()
            )

        except ValueError as error:

            raise ValueError(
                "market_data_date must be 'latest' "
                "or a date in YYYY-MM-DD format."
            ) from error

    # Excel / pandas date.
    elif isinstance(
        value,
        pd.Timestamp,
    ):

        market_data_date = (
            value.date()
        )

    # Python datetime.
    elif isinstance(
        value,
        datetime,
    ):

        market_data_date = (
            value.date()
        )

    # Python date.
    elif isinstance(
        value,
        date,
    ):

        market_data_date = (
            value
        )

    else:

        raise ValueError(
            "market_data_date must be 'latest' "
            "or a date in YYYY-MM-DD format."
        )

    if (
        market_data_date
        > date.today()
    ):
        raise ValueError(
            "market_data_date must not be in the future."
        )

    return market_data_date


# ---------------------------------------------------------
# Output flag helper
# ---------------------------------------------------------

# Excel and pandas may represent TRUE/FALSE values as
# booleans, 1/0, or strings depending on the workbook.
def _is_valid_boolean_setting(
    value,
):

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return True

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return value in {
            0,
            1,
        }

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):

        if not np.isfinite(
            value
        ):
            return False

        return value in {
            0.0,
            1.0,
        }

    if isinstance(
        value,
        str,
    ):

        return (
            value
            .strip()
            .lower()
            in {
                "true",
                "false",
                "yes",
                "no",
                "1",
                "0",
            }
        )

    return False


# ---------------------------------------------------------
# Portfolio validation
# ---------------------------------------------------------

# Validate the basic structure and values of the credit portfolio.
def validate_portfolio(
    portfolio,
):

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
        "maturity",
        "coupon_rate",
        "payment_frequency",
        "sector",
        "region",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in portfolio.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # Exposure must be strictly positive.
    if (
        portfolio[
            "exposure"
        ]
        <= 0
    ).any():

        raise ValueError(
            "Exposure must be strictly positive."
        )

    # LGD must lie between zero and one.
    if (
        (
            portfolio[
                "lgd"
            ]
            < 0
        )
        | (
            portfolio[
                "lgd"
            ]
            > 1
        )
    ).any():

        raise ValueError(
            "LGD must be between 0 and 1."
        )

    # Currency must be provided.
    if portfolio[
        "currency"
    ].isna().any():

        raise ValueError(
            "Currency must be provided for every counterparty."
        )

    if (
        portfolio[
            "currency"
        ]
        .astype(
            str
        )
        .str
        .strip()
        .eq(
            ""
        )
        .any()
    ):

        raise ValueError(
            "Currency must not be empty."
        )

    # PD method must be either rating or Merton.
    valid_pd_methods = {
        "rating",
        "merton",
    }

    if not (
        portfolio[
            "pd_method"
        ]
        .isin(
            valid_pd_methods
        )
        .all()
    ):

        raise ValueError(
            "pd_method must be either "
            "'rating' or 'merton'."
        )

    # Rating counterparties must have a rating.
    rating_rows = (
        portfolio[
            "pd_method"
        ]
        == "rating"
    )

    if (
        portfolio.loc[
            rating_rows,
            "rating",
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "A rating must be provided when "
            "pd_method is 'rating'."
        )

    # Merton counterparties need either a ticker
    # or complete manual company inputs.
    #
    # The risk free rate is not a portfolio input.
    # It is obtained automatically from market data
    # and replaced by the frozen fallback snapshot
    # if retrieval fails.
    merton_rows = (
        portfolio[
            "pd_method"
        ]
        == "merton"
    )

    manual_merton_columns = [
        "equity_value",
        "equity_volatility",
        "debt",
    ]

    for _, row in (
        portfolio.loc[
            merton_rows
        ]
        .iterrows()
    ):

        has_ticker = (
            pd.notna(
                row[
                    "ticker"
                ]
            )
            and str(
                row[
                    "ticker"
                ]
            ).strip()
            != ""
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
                f"Merton inputs are incomplete for "
                f"counterparty "
                f"'{row['counterparty']}'. "
                f"Provide either a ticker or "
                f"complete manual Merton inputs."
            )

    # Unlisted counterparties need a listed proxy
    # for factor loading estimation.
    for _, row in (
        portfolio
        .iterrows()
    ):

        has_ticker = (
            pd.notna(
                row[
                    "ticker"
                ]
            )
            and str(
                row[
                    "ticker"
                ]
            ).strip()
            != ""
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
            ).strip()
            != ""
        )

        if (
            not has_ticker
            and not has_factor_loading_proxy
        ):

            raise ValueError(
                f"A factor loading proxy ticker must "
                f"be provided for unlisted "
                f"counterparty "
                f"'{row['counterparty']}'."
            )

    return True


# ---------------------------------------------------------
# Model settings validation
# ---------------------------------------------------------

# Validate model settings.
def validate_model_settings(
    settings,
):

    required_settings = [
        "number_simulations",
        "seed",
        "confidence_level",
        "dependence_model",
        "t_degrees_of_freedom",
        "factor_structure",
        "factor_lookback_years",
        "base_currency",
        "market_data_date",
        "create_excel_output",
        "create_pdf_output",
        "create_plot_files",
    ]

    missing_settings = [
        setting
        for setting in required_settings
        if setting not in settings
    ]

    if missing_settings:

        raise ValueError(
            f"Missing model settings: "
            f"{missing_settings}"
        )

    # Number of simulations.
    number_simulations = (
        settings[
            "number_simulations"
        ]
    )

    try:

        number_simulations_float = float(
            number_simulations
        )

    except (
        TypeError,
        ValueError,
    ) as error:

        raise ValueError(
            "number_simulations must be "
            "a positive integer."
        ) from error

    if (
        not number_simulations_float.is_integer()
        or number_simulations_float
        <= 0
    ):

        raise ValueError(
            "number_simulations must be "
            "a positive integer."
        )

    # Random seed.
    seed = settings[
        "seed"
    ]

    try:

        seed_float = float(
            seed
        )

    except (
        TypeError,
        ValueError,
    ) as error:

        raise ValueError(
            "seed must be a non-negative integer."
        ) from error

    if (
        not seed_float.is_integer()
        or seed_float
        < 0
    ):

        raise ValueError(
            "seed must be a non-negative integer."
        )

    # Confidence level.
    confidence_level = (
        settings[
            "confidence_level"
        ]
    )

    if not (
        0
        < confidence_level
        < 1
    ):

        raise ValueError(
            "confidence_level must be "
            "between zero and one."
        )

    # Dependence model.
    valid_dependence_models = {
        "independent",
        "gaussian_copula",
        "t_copula",
    }

    dependence_model = (
        settings[
            "dependence_model"
        ]
    )

    if (
        dependence_model
        not in valid_dependence_models
    ):

        raise ValueError(
            "dependence_model must be "
            "'independent', "
            "'gaussian_copula', "
            "or 't_copula'."
        )

    # t-copula degrees of freedom.
    if (
        dependence_model
        == "t_copula"
    ):

        t_degrees_of_freedom = (
            settings[
                "t_degrees_of_freedom"
            ]
        )

        if (
            t_degrees_of_freedom
            <= 2
        ):

            raise ValueError(
                "t_degrees_of_freedom must be "
                "greater than two when "
                "dependence_model is 't_copula'."
            )

    # Factor structure.
    valid_factor_structures = {
        "single_factor",
        "global_sector",
        "global_region",
        "global_sector_region",
    }

    factor_structure = (
        settings[
            "factor_structure"
        ]
    )

    if (
        factor_structure
        not in valid_factor_structures
    ):

        raise ValueError(
            "factor_structure must be "
            "'single_factor', "
            "'global_sector', "
            "'global_region', or "
            "'global_sector_region'."
        )

    # Factor lookback period.
    factor_lookback_years = (
        settings[
            "factor_lookback_years"
        ]
    )

    if (
        factor_lookback_years
        <= 0
    ):

        raise ValueError(
            "factor_lookback_years must be "
            "greater than zero."
        )

    # Portfolio base currency.
    base_currency = (
        str(
            settings[
                "base_currency"
            ]
        )
        .strip()
        .upper()
    )

    if (
        not base_currency
        or len(
            base_currency
        )
        != 3
    ):

        raise ValueError(
            "base_currency must be a "
            "three-letter currency code."
        )

    # Market data date.
    parse_market_data_date(
        settings[
            "market_data_date"
        ]
    )

    # Output flags.
    #
    # Excel / pandas may return TRUE/FALSE cells as
    # bool, 1/0, 1.0/0.0 or strings. All equivalent
    # representations are accepted here.
    output_settings = [
        "create_excel_output",
        "create_pdf_output",
        "create_plot_files",
    ]

    for setting_name in (
        output_settings
    ):

        if not (
            _is_valid_boolean_setting(
                settings[
                    setting_name
                ]
            )
        ):

            raise ValueError(
                f"{setting_name} must be "
                f"TRUE or FALSE."
            )

    return True


# ---------------------------------------------------------
# Factor proxy validation
# ---------------------------------------------------------

# Validate factor proxy mappings.
def validate_factor_proxies(
    factor_proxies,
):

    required_columns = [
        "factor_type",
        "factor_name",
        "ticker",
        "description",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in factor_proxies.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing factor proxy columns: "
            f"{missing_columns}"
        )

    valid_factor_types = {
        "global",
        "region",
        "sector",
    }

    if not (
        factor_proxies[
            "factor_type"
        ]
        .isin(
            valid_factor_types
        )
        .all()
    ):

        raise ValueError(
            "factor_type must be "
            "'global', 'region', or 'sector'."
        )

    if (
        factor_proxies[
            "factor_name"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "factor_name must not be missing."
        )

    if (
        factor_proxies[
            "ticker"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "ticker must not be missing."
        )

    if (
        factor_proxies
        .duplicated(
            subset=[
                "factor_type",
                "factor_name",
            ]
        )
        .any()
    ):

        raise ValueError(
            "Each factor_type and factor_name "
            "combination must be unique."
        )

    number_global_factors = (
        factor_proxies[
            "factor_type"
        ]
        .eq(
            "global"
        )
        .sum()
    )

    if (
        number_global_factors
        != 1
    ):

        raise ValueError(
            "Exactly one global factor "
            "must be defined."
        )

    return True


# ---------------------------------------------------------
# Fallback market data validation
# ---------------------------------------------------------

# Validate fallback market data snapshot.
def validate_market_data(
    market_data,
    base_currency,
):

    required_columns = [
        "data_type",
        "currency",
        "base_currency",
        "tenor_years",
        "value",
        "as_of_date",
        "curve_type",
        "source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in market_data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing market data columns: "
            f"{missing_columns}"
        )

    if market_data.empty:

        raise ValueError(
            "market_data must not be empty."
        )

    data_types = (
        market_data[
            "data_type"
        ]
        .astype(
            str
        )
        .str
        .strip()
        .str
        .lower()
    )

    currencies = (
        market_data[
            "currency"
        ]
        .astype(
            str
        )
        .str
        .strip()
        .str
        .upper()
    )

    model_base_currency = (
        str(
            base_currency
        )
        .strip()
        .upper()
    )

    valid_data_types = {
        "fx",
        "risk_free",
    }

    if not (
        data_types
        .isin(
            valid_data_types
        )
        .all()
    ):

        raise ValueError(
            "data_type must be either "
            "'fx' or 'risk_free'."
        )

    if (
        market_data[
            "currency"
        ]
        .isna()
        .any()
        or currencies
        .eq(
            ""
        )
        .any()
    ):

        raise ValueError(
            "Currency must be provided for "
            "every market data row."
        )

    numeric_values = (
        pd.to_numeric(
            market_data[
                "value"
            ],
            errors="coerce",
        )
    )

    if (
        numeric_values
        .isna()
        .any()
    ):

        raise ValueError(
            "Market data values must be numeric."
        )

    if (
        market_data[
            "as_of_date"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "as_of_date must be provided for "
            "every market data row."
        )

    snapshot_dates = (
        pd.to_datetime(
            market_data[
                "as_of_date"
            ],
            errors="coerce",
        )
    )

    if (
        snapshot_dates
        .isna()
        .any()
    ):

        raise ValueError(
            "as_of_date contains an invalid date."
        )

    if (
        snapshot_dates
        .dt
        .normalize()
        .nunique()
        != 1
    ):

        raise ValueError(
            "All fallback market data must use "
            "the same as_of_date."
        )

    fx_rows = (
        data_types
        == "fx"
    )

    risk_free_rows = (
        data_types
        == "risk_free"
    )

    fx_data = (
        market_data.loc[
            fx_rows
        ]
        .copy()
    )

    risk_free_data = (
        market_data.loc[
            risk_free_rows
        ]
        .copy()
    )

    fx_currencies = (
        currencies.loc[
            fx_rows
        ]
    )

    risk_free_currencies = (
        currencies.loc[
            risk_free_rows
        ]
    )

    # -----------------------------
    # FX fallback validation
    # -----------------------------

    if fx_data.empty:

        raise ValueError(
            "market_data must contain "
            "FX fallback data."
        )

    fx_base_currencies = (
        fx_data[
            "base_currency"
        ]
        .astype(
            str
        )
        .str
        .strip()
        .str
        .upper()
    )

    if (
        fx_data[
            "base_currency"
        ]
        .isna()
        .any()
        or fx_base_currencies
        .eq(
            ""
        )
        .any()
    ):

        raise ValueError(
            "base_currency must be provided "
            "for every FX row."
        )

    unique_snapshot_bases = (
        fx_base_currencies
        .unique()
    )

    if (
        len(
            unique_snapshot_bases
        )
        != 1
    ):

        raise ValueError(
            "All FX fallback rates must use "
            "the same snapshot reference currency."
        )

    snapshot_base_currency = (
        unique_snapshot_bases[
            0
        ]
    )

    fx_values = (
        pd.to_numeric(
            fx_data[
                "value"
            ],
            errors="coerce",
        )
    )

    if (
        fx_values
        <= 0
    ).any():

        raise ValueError(
            "FX fallback rates must be "
            "strictly positive."
        )

    if (
        fx_currencies
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Each currency may appear only once "
            "in FX fallback data."
        )

    if (
        fx_data[
            "tenor_years"
        ]
        .notna()
        .any()
    ):

        raise ValueError(
            "FX fallback rows must not contain "
            "tenor_years."
        )

    snapshot_base_rows = (
        fx_data.loc[
            fx_currencies
            == snapshot_base_currency
        ]
    )

    if (
        len(
            snapshot_base_rows
        )
        != 1
    ):

        raise ValueError(
            "FX fallback data must contain "
            "exactly one row for the snapshot "
            "reference currency."
        )

    snapshot_base_value = float(
        snapshot_base_rows[
            "value"
        ]
        .iloc[
            0
        ]
    )

    if (
        abs(
            snapshot_base_value
            - 1.0
        )
        > 1e-12
    ):

        raise ValueError(
            "The FX rate of the snapshot "
            "reference currency to itself "
            "must equal one."
        )

    if (
        model_base_currency
        not in set(
            fx_currencies
        )
    ):

        raise ValueError(
            f"Model base currency "
            f"'{model_base_currency}' "
            f"is not available in the "
            f"FX fallback snapshot."
        )

    # -----------------------------
    # Risk free fallback validation
    # -----------------------------

    if risk_free_data.empty:

        raise ValueError(
            "market_data must contain "
            "risk free fallback data."
        )

    # Negative risk free rates are allowed.
    risk_free_values = (
        pd.to_numeric(
            risk_free_data[
                "value"
            ],
            errors="coerce",
        )
    )

    if (
        risk_free_values
        .isna()
        .any()
    ):

        raise ValueError(
            "Risk free fallback rates "
            "must be numeric."
        )

    risk_free_tenors = (
        pd.to_numeric(
            risk_free_data[
                "tenor_years"
            ],
            errors="coerce",
        )
    )

    if (
        risk_free_tenors
        .isna()
        .any()
    ):

        raise ValueError(
            "Risk free fallback rows must "
            "contain tenor_years."
        )

    if (
        risk_free_tenors
        <= 0
    ).any():

        raise ValueError(
            "Risk free tenors must be "
            "strictly positive."
        )

    risk_free_keys = (
        pd.DataFrame({
            "currency":
                risk_free_currencies
                .to_numpy(),
            "tenor_years":
                risk_free_tenors
                .to_numpy(),
        })
    )

    if (
        risk_free_keys
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Each currency and tenor combination "
            "must be unique in risk free "
            "fallback data."
        )

    if (
        risk_free_data[
            "curve_type"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "curve_type must be provided "
            "for risk free fallback data."
        )

    if (
        risk_free_data[
            "source"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "source must be provided for "
            "risk free fallback data."
        )

    # flat_policy_proxy rows are simplified
    # fallback inputs only. They are not
    # calibrated market yield curves.

    return True


# ---------------------------------------------------------
# Rating migration validation
# ---------------------------------------------------------

# Validate rating migration matrix and credit spreads.
def validate_rating_migration_matrix(
    rating_migration_matrix,
):

    # Synthetic one year rating transition matrix
    # and spreads. Values are for demonstration
    # purposes only and do not represent data from
    # a rating agency.

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

    missing_columns = [
        column
        for column in required_columns
        if column
        not in rating_migration_matrix.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing rating migration columns: "
            f"{missing_columns}"
        )

    current_ratings = (
        rating_migration_matrix[
            "current_rating"
        ]
        .astype(
            str
        )
        .str
        .upper()
    )

    if (
        set(
            current_ratings
        )
        != set(
            rating_states
        )
    ):

        raise ValueError(
            "The migration matrix must contain "
            "exactly AAA, AA, A, BBB, BB, B, "
            "CCC, and D."
        )

    if (
        current_ratings
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Each current rating must appear "
            "exactly once."
        )

    transition_matrix = (
        rating_migration_matrix[
            rating_states
        ]
    )

    if (
        (
            transition_matrix
            < 0
        )
        | (
            transition_matrix
            > 1
        )
    ).any().any():

        raise ValueError(
            "Transition probabilities must "
            "be between zero and one."
        )

    calculated_row_sums = (
        transition_matrix
        .sum(
            axis=1
        )
    )

    if not (
        calculated_row_sums
        .sub(
            1.0
        )
        .abs()
        .le(
            1e-8
        )
        .all()
    ):

        raise ValueError(
            "Each rating transition row "
            "must sum to one."
        )

    non_default_rows = (
        current_ratings
        != "D"
    )

    if (
        rating_migration_matrix.loc[
            non_default_rows,
            "spread_decimal",
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Every non-default rating must "
            "have a credit spread."
        )

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

    spread_difference = (
        (
            rating_migration_matrix.loc[
                non_default_rows,
                "spread_bps",
            ]
            / 10000
        )
        - rating_migration_matrix.loc[
            non_default_rows,
            "spread_decimal",
        ]
    ).abs()

    if not (
        spread_difference
        .le(
            1e-10
        )
        .all()
    ):

        raise ValueError(
            "spread_decimal must equal "
            "spread_bps / 10000."
        )

    ordered_spreads = (
        rating_migration_matrix
        .assign(
            current_rating=current_ratings
        )
        .set_index(
            "current_rating"
        )
        .loc[
            rating_states[
                :-1
            ],
            "spread_decimal",
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not (
        ordered_spreads[
            1:
        ]
        > ordered_spreads[
            :-1
        ]
    ).all():

        raise ValueError(
            "Credit spreads must increase "
            "as credit quality deteriorates."
        )

    indexed_matrix = (
        rating_migration_matrix
        .assign(
            current_rating=current_ratings
        )
        .set_index(
            "current_rating"
        )
    )

    default_row = (
        indexed_matrix.loc[
            "D",
            rating_states,
        ]
    )

    if (
        default_row[
            "D"
        ]
        != 1
        or default_row
        .drop(
            "D"
        )
        .sum()
        != 0
    ):

        raise ValueError(
            "Default must be an absorbing state."
        )

    return True