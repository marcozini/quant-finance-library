#### Credit Risk Migration ####

from quant_finance.credit.bond_valuation import bond_price


# Synthetic credit spreads by rating.
# Values are annualized spreads in decimal form and are for demonstration only.
RATING_SPREADS = {
    "AAA": 0.0030,
    "AA": 0.0050,
    "A": 0.0080,
    "BBB": 0.0150,
    "BB": 0.0300,
    "B": 0.0600,
    "CCC": 0.1200,
}
# TODO: Move this later to Excel as an Input

# Return the credit spread associated with a rating.
def rating_to_spread(rating):

    rating = rating.upper()

    if rating not in RATING_SPREADS:
        raise ValueError(f"Invalid rating: {rating}")

    return RATING_SPREADS[rating]


# Revalue a bond after a rating migration.
def migrated_bond_value(
    times,
    cash_flows,
    risk_free_rates,
    migrated_rating,
):

    migrated_spread = rating_to_spread(migrated_rating)

    return bond_price(
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        spread=migrated_spread,
    )


# Calculate the loss caused by a rating migration.
def migration_loss(
    current_value,
    times,
    cash_flows,
    risk_free_rates,
    migrated_rating,
):

    migrated_value = migrated_bond_value(
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        migrated_rating=migrated_rating,
    )

    loss = current_value - migrated_value

    return migrated_value, loss