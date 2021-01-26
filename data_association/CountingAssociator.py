from data_association.Associator import Associator

import numpy


class CountingAssociator(Associator):
    """
    Class for the CountingAssociator
    """
    associated = False
    num_consecutive_hits = 0
    num_consecutive_misses = 0

    def __init__(self, association_distance_threshold=10, consecutive_hits_confirm_association=3,
                 consecutive_misses_end_association=2):
        """
        :param association_distance_threshold: A distance used to decide whether two states are considered close enough
            to count as a "hit"
        :param consecutive_hits_confirm_association: Number of consecutive hits needed to confirm an association
        :param consecutive_misses_end_association: Number of consecutive misses needed to end an association
        """
        super().__init__()
        self.association_distance_threshold = association_distance_threshold
        self.consecutive_hits_confirm_association = consecutive_hits_confirm_association
        self.consecutive_misses_end_association = consecutive_misses_end_association

    def associate_tracks(self, tracks1, tracks2):
        """
        Checks for association using the counting technique. The algorithm is described in detail in the report.

        Assumptions: assumes that the method is used each time an association check is performed. It only examines the
        last state in the tracks lists. Another method should be used if one want to evaluate association based on e.g.
        the last 5 states in the tracks list, and only that (i.e., use no other information than the x last states).

        :param tracks1:
        :param tracks2:
        :return: true to the last element of tracks1 and tracks2 are considered associated, and false if they are not
        considered associated
        """
        # calculate the euclidean distance between the last state of the tracks
        distance = numpy.linalg.norm(tracks1[-1].state_vector - tracks2[-1].state_vector)

        # todo: implement the logic specified
        if self.associated:
            # associate if within distance
            if distance < self.association_distance_threshold:
                return True
            # associate if consecutive misses is lower than end association threshold
            elif self.num_consecutive_misses < self.consecutive_misses_end_association:
                self.num_consecutive_misses += 1
                self.num_consecutive_hits = 0
                return True
            # don't associate and reset number of consecutive hits
            # todo: check if this cover all the cases the tracks where previously associated
            else:
                self.associated = False
                self.num_consecutive_misses = 0  # reset as association is cancelled
                self.num_consecutive_hits = 0
                return False
        else:
            if distance < self.association_distance_threshold:
                self.num_consecutive_hits += 1
                self.associated = self.num_consecutive_hits >= self.consecutive_hits_confirm_association
                return self.associated
            else:
                self.num_consecutive_misses += 1  # not really necessary
                self.num_consecutive_hits = 0
