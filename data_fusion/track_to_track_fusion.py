"""track_to_track_fusion

The module fuses two tracks under the error independence assumption. See the report for derivations.
todo
"""
import numpy as np
from stonesoup.types.update import GaussianStateUpdate


def fuse(track1, track2):
    """
    fuses the tracks todo
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
