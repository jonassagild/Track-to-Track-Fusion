"""track_to_track_association

The module tests two tracks for track association. It uses hypothesis testing to decide whether the two tracks are of
the same target. See report for more mathematical derivation.
"""
import numpy as np
from scipy.stats.distributions import chi2


def test_association_independent_tracks(track1, track2, alpha=0.05):
    """
    Checks whether the tracks are from the same target, under the independence assumption
    :param track1: track to check for association
    :param track2: track to check for association
    :param alpha: desired confidence interval
    :return: true if the tracks are from the same target, false else
    """
    delta_estimates = track1.state_vector - track2.state_vector
    error_delta_estimates = delta_estimates  # as the difference of the true states is 0 if it is the same target
    error_delta_estimates_covar = track1.covar + track2.covar  # under the error independence assumption

    d = error_delta_estimates.transpose() @ np.linalg.inv(error_delta_estimates_covar) @ error_delta_estimates

    # 4 degrees of freedom as we have 4 dimensions in the state vector
    d_alpha = chi2.ppf((1 - alpha), df=4)

    # Accept H0 if d <= d_alpha
    return d <= d_alpha


def test_association_dependent_tracks(track1, track2, cross_cov_ij, cross_cov_ji, alpha=0.05):
    """
    checks whether the tracks are from the same target, when the dependence is accounted for.
    :param track1: track to check for association
    :param track2: track to check for association
    :param cross_cov_ij: cross-covariance of the estimation errors. See article
    :param cross_cov_ji:
    :param alpha:desired confidence interval
    :return: true if the tracks are from the same target, false else
    """
    delta_estimates = track1.state_vector - track2.state_vector
    error_delta_estimates = delta_estimates  # as the difference of the true states is 0 if it is the same target
    error_delta_estimates_covar = track1.covar + track2.covar - cross_cov_ij - cross_cov_ji  # under the error
    # independence
    # assumption

    d = error_delta_estimates.transpose() @ np.linalg.inv(error_delta_estimates_covar) @ error_delta_estimates

    # 4 degrees of freedom as we have 4 dimensions in the state vector
    d_alpha = chi2.ppf((1 - alpha), df=4)

    # Accept H0 if d <= d_alpha
    return d <= d_alpha
