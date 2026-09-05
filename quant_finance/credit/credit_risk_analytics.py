#### Portfolio Credit Risk Analytics ####

import numpy as np


# Calculate expected loss for a single counterparty.
def expected_loss(exposure, pd, lgd):

    if exposure < 0:
        raise ValueError(
            "exposure must be greater than or equal to zero."
        )

    if not 0 <= pd <= 1:
        raise ValueError(
            "pd must be between zero and one."
        )

    if not 0 <= lgd <= 1:
        raise ValueError(
            "lgd must be between zero and one."
        )

    return exposure * pd * lgd


# Calculate total expected loss for a portfolio.
def portfolio_expected_loss(
    exposures,
    pds,
    lgds,
):

    if not (
        len(exposures)
        == len(pds)
        == len(lgds)
    ):
        raise ValueError(
            "exposures, pds, and lgds must have the same length."
        )

    return sum(
        expected_loss(
            exposure,
            pd,
            lgd,
        )
        for exposure, pd, lgd in zip(
            exposures,
            pds,
            lgds,
        )
    )


# Change in portfolio risk charge after adding a new exposure.
def incremental_risk(
    old_rc,
    new_rc,
):

    return new_rc - old_rc


# Additional risk charge per unit of added exposure.
def marginal_risk(
    old_rc,
    new_rc,
    added_asset,
):

    if added_asset <= 0:
        raise ValueError(
            "added_asset must be greater than zero."
        )

    return (
        new_rc - old_rc
    ) / added_asset


# Reduction in risk due to diversification.
def diversification_benefit(
    standalone_rc,
    incremental_rc,
):

    return (
        standalone_rc
        - incremental_rc
    )


# Incremental risk relative to standalone risk.
def diversification_ratio(
    standalone_rc,
    incremental_rc,
):

    if standalone_rc <= 0:
        raise ValueError(
            "standalone_rc must be greater than zero."
        )

    return (
        incremental_rc
        / standalone_rc
    )


# Calculate the number of simulated scenarios belonging to the tail.
def tail_scenario_count(
    number_observations,
    confidence_level,
):

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    if (
        not isinstance(
            number_observations,
            (int, np.integer),
        )
        or number_observations <= 0
    ):
        raise ValueError(
            "number_observations must be a positive integer."
        )

    tail_size = (
        (1 - confidence_level)
        * number_observations
    )

    nearest_integer = round(
        tail_size
    )

    # Floating-point arithmetic can turn exact theoretical
    # values such as 5000 into 5000.000000000005.
    # Use a scale-aware tolerance to identify such cases.
    tolerance = (
        10
        * np.finfo(float).eps
        * max(
            1.0,
            abs(tail_size),
        )
    )

    if (
        abs(
            tail_size
            - nearest_integer
        )
        <= tolerance
    ):
        number_tail_scenarios = int(
            nearest_integer
        )

    else:
        number_tail_scenarios = int(
            np.ceil(
                tail_size
            )
        )

    return max(
        1,
        number_tail_scenarios,
    )


# Calculate the number of simulated scenarios belonging to the tail.
def tail_scenario_count(
    number_observations,
    confidence_level,
):

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    if (
        not isinstance(
            number_observations,
            (int, np.integer),
        )
        or number_observations <= 0
    ):
        raise ValueError(
            "number_observations must be a positive integer."
        )

    tail_size = (
        (1 - confidence_level)
        * number_observations
    )

    nearest_integer = round(
        tail_size
    )

    tolerance = (
        10
        * np.finfo(float).eps
        * max(
            1.0,
            abs(tail_size),
        )
    )

    if (
        abs(
            tail_size
            - nearest_integer
        )
        <= tolerance
    ):
        number_tail_scenarios = int(
            nearest_integer
        )

    else:
        number_tail_scenarios = int(
            np.ceil(
                tail_size
            )
        )

    return max(
        1,
        number_tail_scenarios,
    )


# Calculate Value at Risk from a simulated loss distribution.
def value_at_risk(
    losses,
    confidence_level,
):

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    losses = np.asarray(
        losses,
        dtype=float,
    )

    if losses.size == 0:
        raise ValueError(
            "losses must not be empty."
        )

    return float(
        np.quantile(
            losses,
            confidence_level,
        )
    )


# Calculate Expected Shortfall from a simulated loss distribution.
def expected_shortfall(
    losses,
    confidence_level,
):

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    losses = np.asarray(
        losses,
        dtype=float,
    )

    if losses.size == 0:
        raise ValueError(
            "losses must not be empty."
        )

    sorted_losses = np.sort(
        losses
    )

    number_tail_scenarios = (
        tail_scenario_count(
            number_observations=len(
                sorted_losses
            ),
            confidence_level=confidence_level,
        )
    )

    tail_losses = sorted_losses[
        -number_tail_scenarios:
    ]

    return float(
        tail_losses.mean()
    )


# Calculate unexpected loss relative to expected loss.
def unexpected_loss(
    value_at_risk_value,
    expected_loss_value,
):

    return (
        value_at_risk_value
        - expected_loss_value
    )