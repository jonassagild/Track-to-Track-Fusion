"""track_to_track_association

The module tests two tracks for track association. It uses hypothesis testing to decide whether the two tracks are of
the same target.
todo write more
"""
import numpy as np


def test_association(track1, track2):
    """
    Checks whether the tracks are from the same target
    :param track1:
    :param track2:
    :return:
    """
    delta_estimates = track1.state_vector - track2.state_vector
    error_delta_estimates = delta_estimates  # as the difference of the true states is 0 if it is the same target
    error_delta_estimates_covar = track1.covar + track2.covar  # under the error independence assumption

    D = error_delta_estimates.tranpose() * np.linalg.inv(error_delta_estimates_covar) * error_delta_estimates
    # Accept H0 if D <= D_alpha

    alpha = 0.01

    # how many degrees of freedom to use on the chi-squared distribution? 4 as we have 4 dimensions in the state
    # vector

    

    print("test")
    pass


def main():
    # load tracks and run test_association for each track


    pass


if __name__ == "__main__":
    main()
