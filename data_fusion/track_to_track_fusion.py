"""track_to_track_fusion

The module fuses two tracks. See the report for derivations.
"""
import numpy as np
from stonesoup.types.update import GaussianStateUpdate


def fuse_independent_tracks(track1, track2):
    """
    fuses the tracks under the independence assumption
    :param track1:
    :param track2:
    :return:
    """
    P_j = track1.covar  # covar sensor j
    P_i = track2.covar  # covar sensor i
    x_j = track1.state_vector  # posterior sensor j
    x_i = track2.state_vector  # posterior sensor i

    # fusion equation under the error independence assumption
    fused_posterior = P_j @ np.linalg.inv(P_i + P_j) @ x_i + P_i @ np.linalg.inv(P_i + P_j) @ x_j

    # covariance fusion equation
    fused_covar = P_i @ np.linalg.inv(P_i + P_j) @ P_j

    # store fused information in GaussianStateUpdate object
    # fused = GaussianStateUpdate(fused_posterior, fused_covar)
    # todo return an object and not only the posterior and covar
    return fused_posterior, fused_covar


def fuse_dependent_tracks(track1, track2, cross_covar_ij, cross_covar_ji):
    """
    Fuses the tracks accounting for the dependence of the tracks.
    :param track1:
    :param track2:
    :param cross_covar_ij:
    :param cross_covar_ji:
    :return:
    """
    P_i = track1.covar  # covar sensor j
    P_j = track2.covar  # covar sensor i
    x_i = track1.state_vector  # posterior sensor j
    x_j = track2.state_vector  # posterior sensor i
    P_ij = cross_covar_ij
    P_ji = cross_covar_ji

    # fusion equation of dependent tracks
    fused_posterior = x_i + (P_i - P_ij) @ np.linalg.inv(P_i + P_j - P_ij - P_ji) @ (x_j - x_i)

    # covariance fusion equation
    fused_covar = P_i - (P_i - P_ij) @ np.linalg.inv(P_i + P_j - P_ij - P_ji) @ (P_i - P_ji)

    # store fused information in GaussianStateUpdate object
    # fused = GaussianStateUpdate(fused_posterior, fused_covar)
    # todo return an object and not only the posterior and covar
    return fused_posterior, fused_covar


def fuse_independent_track_estimates(t1_cov, t1_mean, t2_cov, t2_mean):
    """
    fuses the tracks under the independence assumption
    :return:
    """
    P_j = t1_cov  # covar sensor j
    P_i = t2_cov  # covar sensor i
    x_j = t1_mean  # posterior sensor j
    x_i = t2_mean  # posterior sensor i

    # fusion equation under the error independence assumption
    fused_posterior = P_j @ np.linalg.inv(P_i + P_j) @ x_i + P_i @ np.linalg.inv(P_i + P_j) @ x_j

    # covariance fusion equation
    fused_covar = P_i @ np.linalg.inv(P_i + P_j) @ P_j

    return fused_posterior, fused_covar
