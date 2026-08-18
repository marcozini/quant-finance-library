#### Counterparty Credit Risk ####

from quant_finance.credit.ratings import rating_to_pd


def get_counterparty_pd(method, rating=None, merton_pd=None):
    """
    Return the probability of default for a counterparty based on
    either an external credit rating or a Merton model estimate.
    """

    method = method.lower()

    if method == "rating":

        if rating is None:
            raise ValueError("A credit rating must be provided.")

        return rating_to_pd(rating)

    elif method == "merton":

        if merton_pd is None:
            raise ValueError("A Merton PD must be provided.")

        if not 0.0 <= merton_pd <= 1.0:
            raise ValueError("Merton PD must be between 0 and 1.")

        return merton_pd

    else:
        raise ValueError("Method must be either 'rating' or 'merton'.")