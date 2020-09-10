"""calc_cross_cov_estimate_error calculates the cross-covariance of the estimation error

See report to find derivation and formula for calculating the cross-covariance of the estimation error.

"""
import numpy as np


def calc_cross_cov_estimate_error(h_i, h_j, kalman_gain_i, kalman_gain_j, f, q, prev_cross_cov):
    """
    Calculates the cross-covariance of the estimation error. See report for description of variables and formula.
    Assumes same transition model for both trackers
    :param prev_cross_cov:
    :param kalman_gain_j:
    :param kalman_gain_i:
    :param h_i:
    :param h_j: 
    :param f: assumes same transition models for both trackers
    :param q: 
    :return:
    """
    # TODO needs refactoring when decided whether to use semantics or mathematical characters. (uses both currently)
    cross_cov = (np.eye(prev_cross_cov.shape[0]) - kalman_gain_i @ h_i) @ np.linalg.inv(f @ prev_cross_cov @ f.T + q) \
                @ (np.eye(prev_cross_cov.shape[0]) - kalman_gain_j @ h_j).T
    return cross_cov


