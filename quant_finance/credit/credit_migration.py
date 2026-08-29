#### Credit Rating Migration ####

import numpy as np


RATING_STATES = [
    "AAA",
    "AA",
    "A",
    "BBB",
    "BB",
    "B",
    "CCC",
    "D",
]


# Create rating-to-spread mapping from Excel input.

def get_rating_spreads(rating_migration_matrix):

    non_default_ratings = RATING_STATES[:-1]

    rating_spreads = {}

    for rating in non_default_ratings:

        row = rating_migration_matrix[
            rating_migration_matrix["current_rating"]
            == rating
        ]

        if len(row) != 1:
            raise ValueError(
                f"Exactly one row must exist for rating '{rating}'."
            )

        rating_spreads[rating] = float(
            row.iloc[0]["spread_decimal"]
        )

    return rating_spreads


# Retrieve credit spread for a rating.

def rating_to_spread(
    rating,
    rating_spreads,
):

    rating = str(rating).upper()

    if rating not in rating_spreads:
        raise ValueError(
            f"No credit spread available for rating '{rating}'."
        )

    return float(
        rating_spreads[rating]
    )


# Infer a synthetic equivalent rating from a counterparty PD.

def pd_to_equivalent_rating(
    pd,
    rating_migration_matrix,
):

    if not 0 <= pd <= 1:
        raise ValueError(
            "PD must be between zero and one."
        )

    non_default_matrix = (
        rating_migration_matrix[
            rating_migration_matrix["current_rating"]
            != "D"
        ]
    )

    default_probabilities = (
        non_default_matrix["D"]
        .to_numpy(
            dtype=float
        )
    )

    ratings = (
        non_default_matrix[
            "current_rating"
        ]
        .astype(str)
        .to_numpy()
    )

    closest_index = np.argmin(
        np.abs(
            default_probabilities
            - pd
        )
    )

    return ratings[
        closest_index
    ]


# Build a transition row that preserves the counterparty's own PD.

def get_adjusted_transition_probabilities(
    current_rating,
    pd,
    rating_migration_matrix,
):

    current_rating = str(
        current_rating
    ).upper()

    if current_rating not in RATING_STATES[:-1]:
        raise ValueError(
            f"Invalid current rating '{current_rating}'."
        )

    if not 0 <= pd <= 1:
        raise ValueError(
            "PD must be between zero and one."
        )

    row = rating_migration_matrix[
        rating_migration_matrix["current_rating"]
        == current_rating
    ]

    if len(row) != 1:
        raise ValueError(
            f"Exactly one transition row must exist "
            f"for rating '{current_rating}'."
        )

    original_probabilities = (
        row.iloc[0][RATING_STATES]
        .to_numpy(
            dtype=float
        )
    )

    original_default_probability = (
        original_probabilities[-1]
    )

    non_default_probabilities = (
        original_probabilities[:-1]
    )

    original_non_default_total = (
        1.0
        - original_default_probability
    )

    if original_non_default_total <= 0:
        raise ValueError(
            "Non-default transition probability must be positive."
        )

    # Preserve the relative shape of the migration distribution,
    # while forcing default probability to equal the model PD.

    adjusted_non_default = (
        non_default_probabilities
        / original_non_default_total
        * (1.0 - pd)
    )

    adjusted_probabilities = np.append(
        adjusted_non_default,
        pd,
    )

    return adjusted_probabilities


# Value fixed-rate bond using risk-free rate plus credit spread.

def bond_value_with_credit_spread(
    face_value,
    coupon_rate,
    maturity,
    payment_frequency,
    risk_free_rate,
    credit_spread,
):

    if face_value <= 0:
        raise ValueError(
            "face_value must be strictly positive."
        )

    if maturity <= 0:
        raise ValueError(
            "maturity must be strictly positive."
        )

    if (
        not float(payment_frequency).is_integer()
        or payment_frequency <= 0
    ):
        raise ValueError(
            "payment_frequency must be a positive integer."
        )

    payment_frequency = int(
        payment_frequency
    )

    number_periods_float = (
        maturity
        * payment_frequency
    )

    number_periods = int(
        round(
            number_periods_float
        )
    )

    if not np.isclose(
        number_periods_float,
        number_periods,
    ):
        raise ValueError(
            "maturity must correspond to a whole "
            "number of coupon periods."
        )

    discount_rate = (
        risk_free_rate
        + credit_spread
    )

    periodic_discount_rate = (
        discount_rate
        / payment_frequency
    )

    coupon_payment = (
        face_value
        * coupon_rate
        / payment_frequency
    )

    periods = np.arange(
        1,
        number_periods + 1,
    )

    cash_flows = np.full(
        number_periods,
        coupon_payment,
        dtype=float,
    )

    cash_flows[-1] += face_value

    discount_factors = (
        1
        + periodic_discount_rate
    ) ** periods

    bond_value = np.sum(
        cash_flows
        / discount_factors
    )

    return float(
        bond_value
    )


# Calculate mark-to-market loss caused by rating migration.

def migration_loss(
    exposure,
    current_rating,
    new_rating,
    coupon_rate,
    maturity,
    payment_frequency,
    risk_free_rate,
    rating_spreads,
):

    if exposure < 0:
        raise ValueError(
            "exposure must be non-negative."
        )

    current_rating = str(
        current_rating
    ).upper()

    new_rating = str(
        new_rating
    ).upper()

    if new_rating == "D":
        raise ValueError(
            "Default must be handled through LGD."
        )

    current_spread = rating_to_spread(
        rating=current_rating,
        rating_spreads=rating_spreads,
    )

    new_spread = rating_to_spread(
        rating=new_rating,
        rating_spreads=rating_spreads,
    )

    current_value = bond_value_with_credit_spread(
        face_value=1.0,
        coupon_rate=coupon_rate,
        maturity=maturity,
        payment_frequency=payment_frequency,
        risk_free_rate=risk_free_rate,
        credit_spread=current_spread,
    )

    new_value = bond_value_with_credit_spread(
        face_value=1.0,
        coupon_rate=coupon_rate,
        maturity=maturity,
        payment_frequency=payment_frequency,
        risk_free_rate=risk_free_rate,
        credit_spread=new_spread,
    )

    relative_loss = (
        current_value
        - new_value
    ) / current_value

    return float(
        exposure
        * relative_loss
    )