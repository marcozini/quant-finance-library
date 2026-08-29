#### Credit Portfolio Simulation ####

import numpy as np
from scipy.stats import norm, t

from quant_finance.credit.credit_migration import (
    RATING_STATES,
    get_rating_spreads,
    pd_to_equivalent_rating,
    get_adjusted_transition_probabilities,
    migration_loss,
)


# Lower latent credit values correspond to worse credit outcomes.
CREDIT_QUALITY_ORDER = ["D", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"]


def _get_active_factors(factor_structure):

    if factor_structure == "single_factor":
        return True, False, False

    if factor_structure == "global_sector":
        return True, False, True

    if factor_structure == "global_region":
        return True, True, False

    if factor_structure == "global_sector_region":
        return True, True, True

    raise ValueError(f"Unsupported factor structure: '{factor_structure}'.")


def _validate_simulation_inputs(portfolio, dependence_model):

    required_columns = ["counterparty", "exposure", "lgd", "pd"]

    if dependence_model != "independent":
        required_columns += [
            "region",
            "sector",
            "global_loading",
            "region_loading",
            "sector_loading",
        ]

    missing_columns = [column for column in required_columns if column not in portfolio.columns]

    if missing_columns:
        raise ValueError(f"Missing simulation columns: {missing_columns}")

    if ((portfolio["pd"] < 0) | (portfolio["pd"] > 1)).any():
        raise ValueError("PD must be between 0 and 1.")

    if (portfolio["exposure"] < 0).any():
        raise ValueError("Exposure must be non-negative.")

    if ((portfolio["lgd"] < 0) | (portfolio["lgd"] > 1)).any():
        raise ValueError("LGD must be between 0 and 1.")

    if dependence_model != "independent":
        loading_columns = ["global_loading", "region_loading", "sector_loading"]

        if portfolio[loading_columns].isna().any().any():
            raise ValueError("Factor loadings must not be missing.")


def _simulate_gaussian_latent_variables(portfolio, number_simulations, factor_structure, rng):

    use_global, use_region, use_sector = _get_active_factors(factor_structure)
    number_counterparties = len(portfolio)

    latent_variables = np.zeros((number_simulations, number_counterparties))

    # One global shock per scenario.
    global_shock = rng.normal(size=number_simulations) if use_global else np.zeros(number_simulations)

    # One regional shock per region and scenario.
    region_shocks = {}

    if use_region:
        for region in portfolio["region"].unique():
            region_shocks[region] = rng.normal(size=number_simulations)

    # One sector shock per sector and scenario.
    sector_shocks = {}

    if use_sector:
        for sector in portfolio["sector"].unique():
            sector_shocks[sector] = rng.normal(size=number_simulations)

    # Independent idiosyncratic shock for every counterparty and scenario.
    idiosyncratic_shocks = rng.normal(size=(number_simulations, number_counterparties))

    for column_index, (_, row) in enumerate(portfolio.iterrows()):

        global_loading = float(row["global_loading"]) if use_global else 0.0
        region_loading = float(row["region_loading"]) if use_region else 0.0
        sector_loading = float(row["sector_loading"]) if use_sector else 0.0

        systematic_variance = global_loading**2 + region_loading**2 + sector_loading**2

        if systematic_variance > 1 + 1e-10:
            raise ValueError(
                f"Systematic factor variance exceeds one for counterparty "
                f"'{row['counterparty']}'."
            )

        idiosyncratic_loading = np.sqrt(max(0.0, 1.0 - systematic_variance))

        latent_variable = global_loading * global_shock

        if use_region:
            latent_variable += region_loading * region_shocks[row["region"]]

        if use_sector:
            latent_variable += sector_loading * sector_shocks[row["sector"]]

        latent_variable += idiosyncratic_loading * idiosyncratic_shocks[:, column_index]

        latent_variables[:, column_index] = latent_variable

    return latent_variables


def simulate_credit_uniforms(portfolio, settings):

    number_simulations = int(settings["number_simulations"])
    seed = int(settings["seed"])
    dependence_model = settings["dependence_model"]
    factor_structure = settings["factor_structure"]

    _validate_simulation_inputs(portfolio, dependence_model)

    rng = np.random.default_rng(seed)

    # Independent counterparties.
    if dependence_model == "independent":
        return rng.uniform(size=(number_simulations, len(portfolio)))

    latent_gaussian = _simulate_gaussian_latent_variables(
        portfolio=portfolio,
        number_simulations=number_simulations,
        factor_structure=factor_structure,
        rng=rng,
    )

    # Gaussian copula.
    if dependence_model == "gaussian_copula":
        return norm.cdf(latent_gaussian)

    # Student-t copula using one shared radial shock per scenario.
    if dependence_model == "t_copula":

        degrees_of_freedom = float(settings["t_degrees_of_freedom"])

        if degrees_of_freedom <= 2:
            raise ValueError("t-copula degrees of freedom must be greater than two.")

        chi_square_scale = (
            rng.chisquare(df=degrees_of_freedom, size=number_simulations)
            / degrees_of_freedom
        )

        latent_t = latent_gaussian / np.sqrt(chi_square_scale)[:, None]

        return t.cdf(latent_t, df=degrees_of_freedom)

    raise ValueError(f"Unsupported dependence model: '{dependence_model}'.")


def simulate_default_matrix(portfolio, settings):

    credit_uniforms = simulate_credit_uniforms(portfolio, settings)
    pds = portfolio["pd"].to_numpy(dtype=float)

    return credit_uniforms < pds


def simulate_portfolio_losses(portfolio, settings, return_defaults=False):

    default_matrix = simulate_default_matrix(portfolio, settings)

    loss_given_default = (
        portfolio["exposure"].to_numpy(dtype=float)
        * portfolio["lgd"].to_numpy(dtype=float)
    )

    portfolio_losses = default_matrix @ loss_given_default

    if return_defaults:
        return portfolio_losses, default_matrix

    return portfolio_losses


def _get_starting_rating(row, rating_migration_matrix):

    if row["pd_method"] == "rating":
        return str(row["rating"]).upper()

    if row["pd_method"] == "merton":
        return pd_to_equivalent_rating(
            pd=float(row["pd"]),
            rating_migration_matrix=rating_migration_matrix,
        )

    raise ValueError(f"Unsupported PD method: '{row['pd_method']}'.")


def simulate_migration_states(portfolio, settings, rating_migration_matrix):

    credit_uniforms = simulate_credit_uniforms(portfolio, settings)

    number_simulations = int(settings["number_simulations"])
    number_counterparties = len(portfolio)

    migration_states = np.empty(
        (number_simulations, number_counterparties),
        dtype=object,
    )

    for column_index, (_, row) in enumerate(portfolio.iterrows()):

        current_rating = _get_starting_rating(row, rating_migration_matrix)

        transition_probabilities = get_adjusted_transition_probabilities(
            current_rating=current_rating,
            pd=float(row["pd"]),
            rating_migration_matrix=rating_migration_matrix,
        )

        # Transition matrix is stored AAA -> D.
        probability_mapping = dict(zip(RATING_STATES, transition_probabilities))

        # Reorder worst -> best so small uniforms correspond to poor credit outcomes.
        ordered_probabilities = np.array(
            [probability_mapping[state] for state in CREDIT_QUALITY_ORDER]
        )

        cumulative_probabilities = np.cumsum(ordered_probabilities)

        state_indices = np.searchsorted(
            cumulative_probabilities,
            credit_uniforms[:, column_index],
            side="right",
        )

        state_indices = np.minimum(
            state_indices,
            len(CREDIT_QUALITY_ORDER) - 1,
        )

        migration_states[:, column_index] = np.array(CREDIT_QUALITY_ORDER)[state_indices]

    return migration_states


def simulate_migration_portfolio_losses(
    portfolio,
    settings,
    rating_migration_matrix,
    return_states=False,
    return_counterparty_losses=False,
):

    migration_states = simulate_migration_states(
        portfolio=portfolio,
        settings=settings,
        rating_migration_matrix=rating_migration_matrix,
    )

    rating_spreads = get_rating_spreads(rating_migration_matrix)

    number_simulations = int(settings["number_simulations"])
    counterparty_losses = np.zeros((number_simulations, len(portfolio)))

    for column_index, (_, row) in enumerate(portfolio.iterrows()):

        current_rating = _get_starting_rating(row, rating_migration_matrix)
        states = migration_states[:, column_index]
        state_losses = {}

        # Pre-calculate loss for every possible destination rating.
        for new_rating in CREDIT_QUALITY_ORDER:

            if new_rating == "D":
                state_losses[new_rating] = float(row["exposure"]) * float(row["lgd"])

            else:
                state_losses[new_rating] = migration_loss(
                    exposure=float(row["exposure"]),
                    current_rating=current_rating,
                    new_rating=new_rating,
                    coupon_rate=float(row["coupon_rate"]),
                    maturity=float(row["maturity"]),
                    payment_frequency=int(row["payment_frequency"]),
                    risk_free_rate=float(row["risk_free_rate"]),
                    rating_spreads=rating_spreads,
                )

        # Assign scenario loss according to simulated rating state.
        for rating_state, loss in state_losses.items():
            counterparty_losses[states == rating_state, column_index] = loss

    portfolio_losses = counterparty_losses.sum(axis=1)

    if return_states and return_counterparty_losses:
        return portfolio_losses, migration_states, counterparty_losses

    if return_states:
        return portfolio_losses, migration_states

    if return_counterparty_losses:
        return portfolio_losses, counterparty_losses

    return portfolio_losses