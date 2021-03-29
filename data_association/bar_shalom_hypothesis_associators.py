"""
contains two classes for the bar-shalom tests. One assuming independence, and one taking the dependence into account
"""
from data_association.track_to_track_association import test_association_dependent_tracks, \
    test_association_independent_tracks


class HypothesisTestIndependenceAssociator():
    """

    """

    def __init__(self, alpha=0.05):
        """

        """
        super().__init__()
        self.alpha = alpha

    def associate_tracks(self, tracks1, tracks2, **kwargs):
        """
        Performs an hypothesis test to check for association
        """
        return test_association_independent_tracks(tracks1[-1], tracks2[-1], alpha=self.alpha)


class HypothesisTestDependenceAssociator:
    """
    Uses the hypothesis test derived by Bar-Shalom to check whether two tracks originate from the same target
    """

    def __init__(self, alpha=0.05):
        """

        """
        super().__init__()
        self.alpha = alpha

    def associate_tracks(self, track1_mean, track1_cov, track2_mean, track2_cov, **kwargs):
        """
        Performs an hypothesis test to check for association
        """
        return test_association_dependent_tracks(track1_mean, track1_cov, track2_mean, track2_cov,
                                                 cross_cov_ij=kwargs['cross_cov_ij'][-1],
                                                 cross_cov_ji=kwargs['cross_cov_ji'][-1], alpha=self.alpha)
