#### Credit Portfolio Model Validation ####

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_finance.credit.credit_risk_analytics import (
    value_at_risk,
    expected_shortfall,
    tail_scenario_count,
)


# Display full tables in the terminal.
pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    180,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

VALIDATION_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
)

ANALYSIS_FOLDER = (
    VALIDATION_FOLDER
    / "summary"
)

RESULT_FILENAME = (
    "credit_portfolio_results.xlsx"
)

BASELINE_FOLDER = (
    "dependence_comparison/t_copula_df5"
)

ANALYSIS_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------

def get_result_path(
    relative_folder,
):

    path = (
        VALIDATION_FOLDER
        / relative_folder
        / RESULT_FILENAME
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Validation result not found: {path}"
        )

    return path


def load_sheet(
    relative_folder,
    sheet_name,
):

    return pd.read_excel(
        get_result_path(
            relative_folder
        ),
        sheet_name=sheet_name,
    )


def load_summary(
    relative_folder,
):

    summary = load_sheet(
        relative_folder,
        "portfolio_summary",
    )

    return dict(
        zip(
            summary["metric"],
            summary["value"],
        )
    )


def load_settings(
    relative_folder,
):

    settings = load_sheet(
        relative_folder,
        "model_settings",
    )

    required_columns = {
        "setting",
        "value",
    }

    if not required_columns.issubset(
        settings.columns
    ):
        raise ValueError(
            "model_settings sheet must contain "
            "'setting' and 'value' columns."
        )

    return dict(
        zip(
            settings["setting"],
            settings["value"],
        )
    )


def load_counterparty_results(
    relative_folder,
):

    return load_sheet(
        relative_folder,
        "counterparty_results",
    )


def load_scenario_losses(
    relative_folder,
):

    scenarios = load_sheet(
        relative_folder,
        "scenario_losses",
    )

    if (
        "portfolio_loss"
        not in scenarios.columns
    ):
        raise ValueError(
            f"'portfolio_loss' missing in "
            f"{relative_folder}"
        )

    return scenarios[
        "portfolio_loss"
    ].to_numpy(
        dtype=float
    )


# ---------------------------------------------------------
# Corrected portfolio risk metrics
# ---------------------------------------------------------

def extract_metrics(
    relative_folder,
):

    summary = load_summary(
        relative_folder
    )

    settings = load_settings(
        relative_folder
    )

    losses = load_scenario_losses(
        relative_folder
    )

    confidence_level = float(
        settings[
            "confidence_level"
        ]
    )

    mean_loss = float(
        losses.mean()
    )

    var = value_at_risk(
        losses=losses,
        confidence_level=confidence_level,
    )

    es = expected_shortfall(
        losses=losses,
        confidence_level=confidence_level,
    )

    return {
        "total_exposure":
            float(
                summary[
                    "Total exposure"
                ]
            ),

        "default_expected_loss":
            float(
                summary[
                    "Default expected loss"
                ]
            ),

        "mean_loss":
            mean_loss,

        "var_99_5":
            var,

        "es_99_5":
            es,

        "unexpected_loss":
            var - mean_loss,

        "maximum_loss":
            float(
                losses.max()
            ),

        "minimum_loss":
            float(
                losses.min()
            ),
    }


# ---------------------------------------------------------
# Experiment-integrity helpers
# ---------------------------------------------------------

def values_match(
    actual,
    expected,
):

    if isinstance(
        expected,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        try:
            return np.isclose(
                float(actual),
                float(expected),
            )

        except (
            TypeError,
            ValueError,
        ):
            return False

    return (
        str(actual)
        .strip()
        .lower()
        ==
        str(expected)
        .strip()
        .lower()
    )


def check_settings(
    relative_folder,
    expected_settings,
):

    settings = load_settings(
        relative_folder
    )

    for (
        key,
        expected,
    ) in expected_settings.items():

        if key not in settings:
            raise ValueError(
                f"Missing setting '{key}' "
                f"in {relative_folder}"
            )

        actual = settings[key]

        if not values_match(
            actual,
            expected,
        ):
            raise ValueError(
                f"Experiment configuration error "
                f"in {relative_folder}: "
                f"{key} = {actual}, "
                f"expected {expected}"
            )


def check_scenario_losses(
    relative_folder,
):

    settings = load_settings(
        relative_folder
    )

    losses = load_scenario_losses(
        relative_folder
    )

    expected_number = int(
        settings[
            "number_simulations"
        ]
    )

    if (
        len(losses)
        != expected_number
    ):
        raise ValueError(
            f"Scenario count mismatch in "
            f"{relative_folder}: "
            f"{len(losses)} != "
            f"{expected_number}"
        )

    if not np.all(
        np.isfinite(
            losses
        )
    ):
        raise ValueError(
            f"Non-finite scenario losses in "
            f"{relative_folder}"
        )


def check_all_lgds(
    relative_folder,
    expected_lgd,
):

    counterparties = (
        load_counterparty_results(
            relative_folder
        )
    )

    lgds = counterparties[
        "lgd"
    ].astype(float)

    if not np.allclose(
        lgds,
        expected_lgd,
    ):
        raise ValueError(
            f"LGD check failed in "
            f"{relative_folder}. "
            f"Expected all LGDs = "
            f"{expected_lgd}"
        )


def get_apple_exposure(
    relative_folder,
):

    counterparties = (
        load_counterparty_results(
            relative_folder
        )
    )

    apple = counterparties[
        counterparties[
            "counterparty"
        ]
        == "Apple"
    ]

    if len(apple) != 1:
        raise ValueError(
            f"Expected exactly one Apple row "
            f"in {relative_folder}"
        )

    return float(
        apple.iloc[0][
            "exposure_local"
        ]
    )


def check_apple_exposure(
    relative_folder,
    expected_exposure,
):

    actual = get_apple_exposure(
        relative_folder
    )

    if not np.isclose(
        actual,
        expected_exposure,
    ):
        raise ValueError(
            f"Apple exposure check failed "
            f"in {relative_folder}: "
            f"{actual} != "
            f"{expected_exposure}"
        )


# ---------------------------------------------------------
# Concentration statistics
# ---------------------------------------------------------

def get_concentration_statistics(
    relative_folder,
):

    counterparties = (
        load_counterparty_results(
            relative_folder
        )
        .copy()
    )

    exposures = counterparties[
        "exposure_base"
    ].astype(float)

    total_exposure = float(
        exposures.sum()
    )

    weights = (
        exposures
        / total_exposure
    )

    apple = counterparties[
        counterparties[
            "counterparty"
        ]
        == "Apple"
    ]

    apple_exposure = float(
        apple.iloc[0][
            "exposure_base"
        ]
    )

    apple_share = (
        apple_exposure
        / total_exposure
    )

    largest_share = float(
        weights.max()
    )

    hhi = float(
        np.sum(
            weights**2
        )
    )

    return {
        "apple_exposure":
            apple_exposure,

        "apple_share_pct":
            apple_share * 100,

        "largest_share_pct":
            largest_share * 100,

        "hhi":
            hhi,
    }


# ---------------------------------------------------------
# Generic comparison-table helper
# ---------------------------------------------------------

def create_comparison_table(
    runs,
    parameter_name,
):

    rows = []

    for (
        run_name,
        parameter_value,
        folder,
    ) in runs:

        rows.append({
            "run":
                run_name,

            parameter_name:
                parameter_value,

            **extract_metrics(
                folder
            ),
        })

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------
# Baseline settings
# ---------------------------------------------------------

BASELINE_SETTINGS = {
    "number_simulations":
        100000,

    "seed":
        0,

    "confidence_level":
        0.995,

    "dependence_model":
        "t_copula",

    "t_degrees_of_freedom":
        5,

    "factor_structure":
        "global_sector_region",

    "base_currency":
        "USD",
}


# ---------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------

DEPENDENCE_RUNS = [
    (
        "Independent",
        "independent",
        "dependence_comparison/independent",
    ),
    (
        "Gaussian copula",
        "gaussian_copula",
        "dependence_comparison/gaussian_copula",
    ),
    (
        "t-copula (df=5)",
        "t_copula",
        "dependence_comparison/t_copula_df5",
    ),
]


DF_RUNS = [
    (
        "df = 3",
        3,
        "sensitivity/t_degrees_of_freedom/t_df3",
    ),
    (
        "df = 5",
        5,
        "dependence_comparison/t_copula_df5",
    ),
    (
        "df = 10",
        10,
        "sensitivity/t_degrees_of_freedom/t_df10",
    ),
]


LGD_RUNS = [
    (
        "LGD = 40%",
        0.40,
        "sensitivity/lgd/lgd_40",
    ),
    (
        "LGD = 50%",
        0.50,
        "dependence_comparison/t_copula_df5",
    ),
    (
        "LGD = 60%",
        0.60,
        "sensitivity/lgd/lgd_60",
    ),
]


CONCENTRATION_RUNS = [
    (
        "Apple = 100",
        100,
        "sensitivity/concentration/apple_100",
    ),
    (
        "Apple = 400",
        400,
        "dependence_comparison/t_copula_df5",
    ),
    (
        "Apple = 800",
        800,
        "sensitivity/concentration/apple_800",
    ),
]


FACTOR_RUNS = [
    (
        "Single factor",
        "single_factor",
        "sensitivity/factor_structure/single_factor",
    ),
    (
        "Global + sector",
        "global_sector",
        "sensitivity/factor_structure/global_sector",
    ),
    (
        "Global + region",
        "global_region",
        "sensitivity/factor_structure/global_region",
    ),
    (
        "Global + sector + region",
        "global_sector_region",
        "dependence_comparison/t_copula_df5",
    ),
]


MONTE_CARLO_RUNS = [
    (
        10000,
        0,
        "monte_carlo_stability/n10000_seed0",
    ),
    (
        10000,
        1,
        "monte_carlo_stability/n10000_seed1",
    ),
    (
        10000,
        2,
        "monte_carlo_stability/n10000_seed2",
    ),

    (
        50000,
        0,
        "monte_carlo_stability/n50000_seed0",
    ),
    (
        50000,
        1,
        "monte_carlo_stability/n50000_seed1",
    ),
    (
        50000,
        2,
        "monte_carlo_stability/n50000_seed2",
    ),

    (
        100000,
        0,
        "dependence_comparison/t_copula_df5",
    ),
    (
        100000,
        1,
        "monte_carlo_stability/n100000_seed1",
    ),
    (
        100000,
        2,
        "monte_carlo_stability/n100000_seed2",
    ),

    (
        250000,
        0,
        "monte_carlo_stability/n250000_seed0",
    ),
    (
        250000,
        1,
        "monte_carlo_stability/n250000_seed1",
    ),
    (
        250000,
        2,
        "monte_carlo_stability/n250000_seed2",
    ),
]


# ---------------------------------------------------------
# Experiment-integrity checks
# ---------------------------------------------------------

def validate_experiments():

    registry = []

    checked_scenario_folders = set()

    def register(
        category,
        run_name,
        folder,
    ):

        if (
            folder
            not in checked_scenario_folders
        ):
            check_scenario_losses(
                folder
            )

            checked_scenario_folders.add(
                folder
            )

        registry.append({
            "category":
                category,

            "run":
                run_name,

            "folder":
                folder,

            "status":
                "OK",
        })

    # Dependence models.
    for (
        name,
        dependence_model,
        folder,
    ) in DEPENDENCE_RUNS:

        expected = {
            **BASELINE_SETTINGS,
            "dependence_model":
                dependence_model,
        }

        check_settings(
            folder,
            expected,
        )

        register(
            "dependence",
            name,
            folder,
        )

    # Degrees of freedom.
    for (
        name,
        df,
        folder,
    ) in DF_RUNS:

        expected = {
            **BASELINE_SETTINGS,
            "t_degrees_of_freedom":
                df,
        }

        check_settings(
            folder,
            expected,
        )

        register(
            "t_df",
            name,
            folder,
        )

    # LGD.
    for (
        name,
        lgd,
        folder,
    ) in LGD_RUNS:

        check_settings(
            folder,
            BASELINE_SETTINGS,
        )

        check_all_lgds(
            folder,
            lgd,
        )

        register(
            "lgd",
            name,
            folder,
        )

    # Concentration.
    for (
        name,
        apple_exposure,
        folder,
    ) in CONCENTRATION_RUNS:

        check_settings(
            folder,
            BASELINE_SETTINGS,
        )

        check_all_lgds(
            folder,
            0.50,
        )

        check_apple_exposure(
            folder,
            apple_exposure,
        )

        register(
            "concentration",
            name,
            folder,
        )

    # Factor structure.
    for (
        name,
        factor_structure,
        folder,
    ) in FACTOR_RUNS:

        expected = {
            **BASELINE_SETTINGS,
            "factor_structure":
                factor_structure,
        }

        check_settings(
            folder,
            expected,
        )

        register(
            "factor_structure",
            name,
            folder,
        )

    # Monte Carlo stability.
    for (
        number_simulations,
        seed,
        folder,
    ) in MONTE_CARLO_RUNS:

        expected = {
            **BASELINE_SETTINGS,

            "number_simulations":
                number_simulations,

            "seed":
                seed,
        }

        check_settings(
            folder,
            expected,
        )

        register(
            "monte_carlo",
            (
                f"n={number_simulations}, "
                f"seed={seed}"
            ),
            folder,
        )

    # Concentration experiments keep total
    # base-currency exposure approximately fixed.
    baseline_total = (
        extract_metrics(
            BASELINE_FOLDER
        )[
            "total_exposure"
        ]
    )

    for folder in [
        "sensitivity/concentration/apple_100",
        "sensitivity/concentration/apple_800",
    ]:

        total_exposure = (
            extract_metrics(
                folder
            )[
                "total_exposure"
            ]
        )

        if not np.isclose(
            total_exposure,
            baseline_total,
            rtol=0.002,
            atol=0.25,
        ):
            raise ValueError(
                "Concentration experiment does not "
                "keep total portfolio exposure "
                "approximately constant: "
                f"{folder}"
            )

    return pd.DataFrame(
        registry
    )


# ---------------------------------------------------------
# Confidence-level sensitivity
# ---------------------------------------------------------

def create_confidence_sensitivity():

    losses = load_scenario_losses(
        BASELINE_FOLDER
    )

    confidence_levels = [
        0.95,
        0.99,
        0.995,
        0.999,
    ]

    rows = []

    for confidence_level in confidence_levels:

        var = value_at_risk(
            losses=losses,
            confidence_level=confidence_level,
        )

        es = expected_shortfall(
            losses=losses,
            confidence_level=confidence_level,
        )

        tail_scenarios = (
            tail_scenario_count(
                number_observations=len(
                    losses
                ),
                confidence_level=confidence_level,
            )
        )

        rows.append({
            "confidence_level":
                confidence_level,

            "confidence_pct":
                confidence_level * 100,

            "var":
                var,

            "expected_shortfall":
                es,

            "tail_scenarios":
                tail_scenarios,
        })

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------
# Monte Carlo stability
# ---------------------------------------------------------

def create_monte_carlo_tables():

    rows = []

    for (
        number_simulations,
        seed,
        folder,
    ) in MONTE_CARLO_RUNS:

        rows.append({
            "number_simulations":
                number_simulations,

            "seed":
                seed,

            **extract_metrics(
                folder
            ),
        })

    raw = pd.DataFrame(
        rows
    )

    summary = (
        raw
        .groupby(
            "number_simulations",
            as_index=False,
        )
        .agg(
            mean_loss_mean=(
                "mean_loss",
                "mean",
            ),

            mean_loss_std=(
                "mean_loss",
                "std",
            ),

            var_mean=(
                "var_99_5",
                "mean",
            ),

            var_std=(
                "var_99_5",
                "std",
            ),

            es_mean=(
                "es_99_5",
                "mean",
            ),

            es_std=(
                "es_99_5",
                "std",
            ),
        )
    )

    summary[
        "var_relative_std_pct"
    ] = (
        summary["var_std"]
        / summary["var_mean"]
        * 100
    )

    summary[
        "es_relative_std_pct"
    ] = (
        summary["es_std"]
        / summary["es_mean"]
        * 100
    )

    return (
        raw,
        summary,
    )


# ---------------------------------------------------------
# Concentration sensitivity
# ---------------------------------------------------------

def create_concentration_table():

    rows = []

    for (
        run_name,
        apple_exposure,
        folder,
    ) in CONCENTRATION_RUNS:

        concentration = (
            get_concentration_statistics(
                folder
            )
        )

        metrics = extract_metrics(
            folder
        )

        row = {
            "run":
                run_name,

            "apple_exposure_input":
                apple_exposure,

            **concentration,

            **metrics,
        }

        row[
            "var_pct_total_exposure"
        ] = (
            row["var_99_5"]
            / row["total_exposure"]
            * 100
        )

        row[
            "es_pct_total_exposure"
        ] = (
            row["es_99_5"]
            / row["total_exposure"]
            * 100
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------

def save_bar_plot(
    table,
    label_column,
    title,
    filename,
):

    plot_data = (
        table
        .set_index(
            label_column
        )[
            [
                "var_99_5",
                "es_99_5",
            ]
        ]
    )

    ax = plot_data.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_title(
        title
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_xlabel(
        ""
    )

    ax.legend(
        [
            "99.5% VaR",
            "99.5% ES",
        ]
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        ANALYSIS_FOLDER
        / filename,
        dpi=200,
    )

    plt.close()


def save_line_plot(
    table,
    x_column,
    title,
    x_label,
    filename,
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        table[x_column],
        table["var_99_5"],
        marker="o",
        label="99.5% VaR",
    )

    ax.plot(
        table[x_column],
        table["es_99_5"],
        marker="o",
        label="99.5% ES",
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        x_label
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        ANALYSIS_FOLDER
        / filename,
        dpi=200,
    )

    plt.close()


# ---------------------------------------------------------
# Main validation analysis
# ---------------------------------------------------------

def main():

    experiment_registry = (
        validate_experiments()
    )

    print(
        "\nEXPERIMENT INTEGRITY CHECKS: "
        "ALL PASSED"
    )


    dependence_comparison = (
        create_comparison_table(
            DEPENDENCE_RUNS,
            "dependence_model",
        )
    )


    df_comparison = (
        create_comparison_table(
            DF_RUNS,
            "degrees_of_freedom",
        )
    )


    lgd_comparison = (
        create_comparison_table(
            LGD_RUNS,
            "lgd",
        )
    )


    concentration_comparison = (
        create_concentration_table()
    )


    factor_comparison = (
        create_comparison_table(
            FACTOR_RUNS,
            "factor_structure",
        )
    )


    (
        mc_raw,
        mc_summary,
    ) = create_monte_carlo_tables()


    confidence_sensitivity = (
        create_confidence_sensitivity()
    )


    # -----------------------------------------------------
    # Internal 99.5% consistency check.
    # -----------------------------------------------------

    baseline_metrics = (
        extract_metrics(
            BASELINE_FOLDER
        )
    )

    confidence_995 = (
        confidence_sensitivity[
            np.isclose(
                confidence_sensitivity[
                    "confidence_level"
                ],
                0.995,
            )
        ]
        .iloc[0]
    )

    var_difference = (
        confidence_995["var"]
        - baseline_metrics[
            "var_99_5"
        ]
    )

    es_difference = (
        confidence_995[
            "expected_shortfall"
        ]
        - baseline_metrics[
            "es_99_5"
        ]
    )


    # -----------------------------------------------------
    # Save consolidated validation workbook.
    # -----------------------------------------------------

    summary_workbook = (
        ANALYSIS_FOLDER
        / "credit_validation_summary.xlsx"
    )

    with pd.ExcelWriter(
        summary_workbook,
        engine="openpyxl",
    ) as writer:

        experiment_registry.to_excel(
            writer,
            sheet_name="experiment_registry",
            index=False,
        )

        dependence_comparison.to_excel(
            writer,
            sheet_name="dependence",
            index=False,
        )

        df_comparison.to_excel(
            writer,
            sheet_name="t_df_sensitivity",
            index=False,
        )

        lgd_comparison.to_excel(
            writer,
            sheet_name="lgd_sensitivity",
            index=False,
        )

        concentration_comparison.to_excel(
            writer,
            sheet_name="concentration",
            index=False,
        )

        factor_comparison.to_excel(
            writer,
            sheet_name="factor_structure",
            index=False,
        )

        mc_raw.to_excel(
            writer,
            sheet_name="mc_stability_raw",
            index=False,
        )

        mc_summary.to_excel(
            writer,
            sheet_name="mc_stability_summary",
            index=False,
        )

        confidence_sensitivity.to_excel(
            writer,
            sheet_name="confidence_sensitivity",
            index=False,
        )


    # -----------------------------------------------------
    # Dependence plot.
    # -----------------------------------------------------

    save_bar_plot(
        dependence_comparison,
        "run",
        "Dependence Model Comparison",
        "dependence_comparison.png",
    )


    # -----------------------------------------------------
    # t degrees-of-freedom plot.
    # -----------------------------------------------------

    save_line_plot(
        df_comparison,
        "degrees_of_freedom",
        "t-Copula Degrees-of-Freedom Sensitivity",
        "Degrees of freedom",
        "t_df_sensitivity.png",
    )


    # -----------------------------------------------------
    # LGD plot.
    # -----------------------------------------------------

    save_line_plot(
        lgd_comparison,
        "lgd",
        "LGD Sensitivity",
        "LGD",
        "lgd_sensitivity.png",
    )


    # -----------------------------------------------------
    # Concentration plot.
    # -----------------------------------------------------

    save_line_plot(
        concentration_comparison,
        "apple_share_pct",
        "Portfolio Concentration Sensitivity",
        "Apple share of portfolio exposure (%)",
        "concentration_sensitivity.png",
    )


    # -----------------------------------------------------
    # Factor-structure plot.
    # -----------------------------------------------------

    save_bar_plot(
        factor_comparison,
        "run",
        "Factor Structure Comparison",
        "factor_structure_comparison.png",
    )


    # -----------------------------------------------------
    # Monte Carlo stability plot.
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.errorbar(
        mc_summary[
            "number_simulations"
        ],
        mc_summary[
            "var_mean"
        ],
        yerr=mc_summary[
            "var_std"
        ],
        marker="o",
        capsize=4,
        label="99.5% VaR",
    )

    ax.errorbar(
        mc_summary[
            "number_simulations"
        ],
        mc_summary[
            "es_mean"
        ],
        yerr=mc_summary[
            "es_std"
        ],
        marker="o",
        capsize=4,
        label="99.5% ES",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_title(
        "Monte Carlo Stability Across Seeds"
    )

    ax.set_xlabel(
        "Number of simulations"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        ANALYSIS_FOLDER
        / "monte_carlo_stability.png",
        dpi=200,
    )

    plt.close()


    # -----------------------------------------------------
    # Confidence-level sensitivity plot.
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        confidence_sensitivity[
            "confidence_pct"
        ],
        confidence_sensitivity[
            "var"
        ],
        marker="o",
        label="VaR",
    )

    ax.plot(
        confidence_sensitivity[
            "confidence_pct"
        ],
        confidence_sensitivity[
            "expected_shortfall"
        ],
        marker="o",
        label="Expected Shortfall",
    )

    ax.set_title(
        "Confidence-Level Sensitivity"
    )

    ax.set_xlabel(
        "Confidence level (%)"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        ANALYSIS_FOLDER
        / "confidence_level_sensitivity.png",
        dpi=200,
    )

    plt.close()


    # -----------------------------------------------------
    # Terminal output.
    # -----------------------------------------------------

    print(
        "\nDEPENDENCE-MODEL COMPARISON"
    )
    print(
        dependence_comparison
    )


    print(
        "\nT-COPULA DEGREES-OF-FREEDOM SENSITIVITY"
    )
    print(
        df_comparison
    )


    print(
        "\nLGD SENSITIVITY"
    )
    print(
        lgd_comparison
    )


    print(
        "\nCONCENTRATION SENSITIVITY"
    )
    print(
        concentration_comparison
    )


    print(
        "\nFACTOR-STRUCTURE COMPARISON"
    )
    print(
        factor_comparison
    )


    print(
        "\nMONTE CARLO STABILITY — RAW RUNS"
    )
    print(
        mc_raw
    )


    print(
        "\nMONTE CARLO STABILITY — ACROSS SEEDS"
    )
    print(
        mc_summary
    )


    print(
        "\nCONFIDENCE-LEVEL SENSITIVITY"
    )
    print(
        confidence_sensitivity
    )


    print(
        "\n99.5% CONFIDENCE CALCULATION CHECK"
    )

    print(
        "VaR difference vs baseline:",
        var_difference,
    )

    print(
        "ES difference vs baseline:",
        es_difference,
    )

    if (
        not np.isclose(
            var_difference,
            0.0,
            atol=1e-10,
        )
        or
        not np.isclose(
            es_difference,
            0.0,
            atol=1e-10,
        )
    ):
        raise ValueError(
            "Confidence-level risk metrics "
            "are inconsistent with baseline."
        )

    print(
        "VaR and ES calculations are fully "
        "consistent with the core risk "
        "analytics functions."
    )


    print(
        "\nValidation summary created:"
    )

    print(
        summary_workbook
    )


    print(
        "\nValidation plots created in:"
    )

    print(
        ANALYSIS_FOLDER
    )


if __name__ == "__main__":
    main()