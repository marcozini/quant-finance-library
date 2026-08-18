#### Portfolio Credit Risk Analytics ####


# Change in portfolio risk charge after adding a new exposure.
def incremental_risk(old_rc, new_rc):

    return new_rc - old_rc


# Additional risk charge per unit of added exposure.
def marginal_risk(old_rc, new_rc, added_asset):

    if added_asset <= 0:
        raise ValueError("added_asset must be greater than zero.")

    return (new_rc - old_rc) / added_asset


# Reduction in risk due to diversification.
def diversification_benefit(standalone_rc, incremental_rc):

    return standalone_rc - incremental_rc


# Incremental risk relative to standalone risk.
def diversification_ratio(standalone_rc, incremental_rc):

    if standalone_rc <= 0:
        raise ValueError("standalone_rc must be greater than zero.")

    return incremental_rc / standalone_rc