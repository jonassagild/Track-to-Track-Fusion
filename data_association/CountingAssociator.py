from data_association.Associator import Associator

import numpy


class CountingAssociator:
    """
    Class for the CountingAssociator
    """

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
        # self.associated = False
        # self.num_consecutive_hits = 0
        # self.num_consecutive_misses = 0

    def associate_tracks(self, mean_track1, mean_track2, association_info, **kwargs):
        """
        Checks for association using the counting technique. The algorithm is described in detail in the report.

        Assumptions: assumes that the method is used each time an association check is performed. It only examines the
        last state in the tracks lists. Another method should be used if one want to evaluate association based on e.g.
        the last 5 states in the tracks list, and only that (i.e., use no other information than the x last states).

        :param mean_track1: array(4,1)
        :param mean_track2: array(4,1)
        :param association_info: object of type CountingAssociationInfo
        :return: true to the last element of tracks1 and tracks2 are considered associated, and false if they are not
        considered associated
        """
        # calculate the euclidean distance between the last state of the tracks
        distance = numpy.linalg.norm(mean_track1 - mean_track2)
        association_info.euclidean_distance = distance

        # todo: implement the logic specified
        if association_info.associated:
            # associate if within distance
            if distance < self.association_distance_threshold:
                return association_info
            # associate if consecutive misses is lower than end association threshold
            elif association_info.num_consecutive_misses < self.consecutive_misses_end_association:
                association_info.num_consecutive_misses += 1
                association_info.num_consecutive_hits = 0
                return association_info
            # don't associate and reset number of consecutive hits
            else:
                association_info.associated = False
                association_info.num_consecutive_misses = 0  # reset as association is cancelled
                association_info.num_consecutive_hits = 0
                return association_info
        else:
            if distance < self.association_distance_threshold:
                association_info.num_consecutive_hits += 1
                association_info.num_consecutive_misses = 0
                association_info.associated = association_info.num_consecutive_hits >= self.\
                    consecutive_hits_confirm_association
                return association_info
            else:
                association_info.num_consecutive_misses += 1  # not really necessary
                association_info.num_consecutive_hits = 0
                return association_info
