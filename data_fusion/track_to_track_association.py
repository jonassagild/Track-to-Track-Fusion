"""track_to_track_association

The module tests two tracks for track association. It uses hypothesis testing to decide whether the two tracks are of
the same target. See report for more mathematical derivation.
"""
import numpy as np
from scipy.stats.distributions import chi2


def test_association_independent_tracks(track1, track2, alpha):
    """
    Checks whether the tracks are from the same target
    :param track1: track to check for association
    :param track2: track to check for association
    :param alpha: desired confidence interval
    :return:
    """
    delta_estimates = track1.state_vector - track2.state_vector
    error_delta_estimates = delta_estimates  # as the difference of the true states is 0 if it is the same target
    error_delta_estimates_covar = track1.covar + track2.covar  # under the error independence assumption

    d = error_delta_estimates.transpose() @ np.linalg.inv(error_delta_estimates_covar) @ error_delta_estimates

    # 4 degrees of freedom as we have 4 dimensions in the state vector
    d_alpha = chi2.ppf((1 - alpha), df=4)

    # Accept H0 if d <= d_alpha
    return d <= d_alpha
