"""
contains two classes for the bar-shalom tests. One assuming independence, and one taking the dependence into account
"""
from data_association.Associator import Associator
from data_association.track_to_track_association import test_association_dependent_tracks, \
    test_association_independent_tracks


class HypothesisTestIndependenceAssociator(Associator):
    """

    """

    def __init__(self):
        """

        """
        super().__init__()

    def associate_tracks(self, tracks1, tracks2, **kwargs):
        """
        Performs an hypothesis test to check for association
        """
        return test_association_independent_tracks(tracks1[-1], tracks2[-1])


class HypothesisTestDependenceAssociator(Associator):
    """

    """

    def __init__(self):
        """

        """
        super().__init__()

    def associate_tracks(self, tracks1, tracks2, **kwargs):
        """
        Performs an hypothesis test to check for association
        """
        return test_association_dependent_tracks(tracks1[-1], tracks2[-1], cross_cov_ij=kwargs['cross_cov_ij'],
                                                 cross_cov_ji=kwargs['cross_cov_ji'])
