#### Ratings ####

# Synthetic one-year credit rating transition matrix.
# Values are probabilities and are used for demonstration purposes only.
# The matrix does not represent data from any rating agency.
#TODO use Excel as input source
TRANSITION_MATRIX = {
    "AAA": {"AAA": 0.9150, "AA": 0.0750, "A": 0.0080, "BBB": 0.0015,
            "BB": 0.0003, "B": 0.0001, "CCC": 0.0000, "D": 0.0001},

    "AA": {"AAA": 0.0080, "AA": 0.9050, "A": 0.0750, "BBB": 0.0090,
           "BB": 0.0015, "B": 0.0005, "CCC": 0.0002, "D": 0.0008},

    "A": {"AAA": 0.0005, "AA": 0.0140, "A": 0.9100, "BBB": 0.0650,
          "BB": 0.0070, "B": 0.0015, "CCC": 0.0005, "D": 0.0015},

    "BBB": {"AAA": 0.0000, "AA": 0.0010, "A": 0.0200, "BBB": 0.9100,
            "BB": 0.0520, "B": 0.0100, "CCC": 0.0030, "D": 0.0040},

    "BB": {"AAA": 0.0000, "AA": 0.0002, "A": 0.0020, "BBB": 0.0300,
           "BB": 0.8650, "B": 0.0700, "CCC": 0.0150, "D": 0.0178},

    "B": {"AAA": 0.0000, "AA": 0.0000, "A": 0.0005, "BBB": 0.0030,
          "BB": 0.0400, "B": 0.8200, "CCC": 0.0800, "D": 0.0565},

    "CCC": {"AAA": 0.0000, "AA": 0.0000, "A": 0.0000, "BBB": 0.0005,
            "BB": 0.0050, "B": 0.0500, "CCC": 0.6500, "D": 0.2945},

    "D": {"AAA": 0.0000, "AA": 0.0000, "A": 0.0000, "BBB": 0.0000,
          "BB": 0.0000, "B": 0.0000, "CCC": 0.0000, "D": 1.0000},
}


def rating_to_pd(rating):
    """
    Return the one-year probability of default for a given credit rating.
    """

    rating = rating.upper()

    if rating not in TRANSITION_MATRIX:
        raise ValueError(f"Unknown credit rating: {rating}")

    return TRANSITION_MATRIX[rating]["D"]