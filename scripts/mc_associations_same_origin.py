"""MC_COUNTING_TECHNIQUE

Script to

"""

# seeds
import numpy as np
from stonesoup.types.state import GaussianState

from data_association.CountingAssociator import CountingAssociator
from data_association.bar_shalom_hypothesis_associators import HypothesisTestIndependenceAssociator, \
    HypothesisTestDependenceAssociator
from trackers.kf_dependent_fusion_async_sensors import KalmanFilterDependentFusionAsyncSensors
from utils import open_object
from utils.scenario_generator import generate_scenario_3

start_seed = 0
end_seed = 5  # normally 500
num_mc_iterations = end_seed - start_seed

# params
save_fig = False

# scenario parameters
sigma_process_list = [0.3]  # [0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 3, 3, 3]
sigma_meas_radar_list = [50]  # [5, 30, 200, 5, 30, 200, 5, 30, 200]
sigma_meas_ais_list = [10]  # [10] * 9
radar_meas_rate = 1  # relevant radar meas rates: 1
ais_meas_rate_list = [6]  # relevant AIS meas rates: 2 - 12
timesteps = 200

# associator params
association_distance_threshold = 10
consecutive_hits_confirm_association = 3
consecutive_misses_end_association = 2
alpha = 0.95

# dicts to store final results for printing in a latex friendly way
Pc_overall = {}  # Pc is the percentage of correctly associating tracks that originate from the same target
something_else_overall = {}
stats = []

for sigma_process, sigma_meas_radar, sigma_meas_ais, ais_meas_rate in zip(sigma_process_list, sigma_meas_radar_list,
                                                                          sigma_meas_ais_list, ais_meas_rate_list):
    for seed in range(start_seed, end_seed):
        # generate scenario
        generate_scenario_3(seed=seed, permanent_save=False, radar_meas_rate=radar_meas_rate,
                            ais_meas_rate=ais_meas_rate, sigma_process=sigma_process,
                            sigma_meas_radar=sigma_meas_radar, sigma_meas_ais=sigma_meas_ais,
                            timesteps=timesteps)

        folder = "temp"  # temp instead of seed, as it is not a permanent save

        # load ground truth and the measurements
        data_folder = "../scenarios/scenario3/" + folder + "/"
        ground_truth = open_object.open_object(data_folder + "ground_truth.pk1")
        measurements_radar = open_object.open_object(data_folder + "measurements_radar.pk1")
        measurements_ais = open_object.open_object(data_folder + "measurements_ais.pk1")

        # load start_time
        start_time = open_object.open_object(data_folder + "start_time.pk1")

        # prior
        initial_covar = np.diag([sigma_meas_radar * sigma_meas_ais, sigma_meas_radar * sigma_process,
                                 sigma_meas_radar * sigma_meas_ais, sigma_meas_radar * sigma_process]) ** 2
        prior = GaussianState([1, 1.1, -1, 0.9], initial_covar, timestamp=start_time)

        kf_dependent_fusion = KalmanFilterDependentFusionAsyncSensors(start_time, prior,
                                                                      sigma_process_radar=sigma_process,
                                                                      sigma_process_ais=sigma_process,
                                                                      sigma_meas_radar=sigma_meas_radar,
                                                                      sigma_meas_ais=sigma_meas_ais)

        tracks_fused_dependent, tracks_radar, tracks_ais = kf_dependent_fusion.track_async(
            start_time, measurements_radar, measurements_ais, fusion_rate=1)

        # use the CountingAssociator to evaluate whether the tracks are associated
        associator = CountingAssociator(association_distance_threshold, consecutive_hits_confirm_association,
                                        consecutive_misses_end_association)
        independence_test_associator = HypothesisTestIndependenceAssociator()
        dependence_test_associator = HypothesisTestDependenceAssociator()

        num_correct_associations_counting = 0
        num_false_mis_associations_counting = 0
        num_correct_associations_independent_hypothesis = 0
        num_false_mis_associations_independent_hypothesis = 0
        num_correct_associations_dependent_hypothesis = 0
        num_false_mis_associations_dependent_hypothesis = 0
        for i in range(1, len(tracks_radar)):
            # use the associators to check for association
            associated_counting = associator.associate_tracks(tracks_radar[:i], tracks_ais[:i])
            associated_independence_test = independence_test_associator.associate_tracks(tracks_radar[:i + 1],
                                                                                         tracks_ais[:i + 1])
            cross_cov_list = kf_dependent_fusion.cross_cov_list
            cross_cov_dict = {'cross_cov_ij': cross_cov_list,
                              'cross_cov_ji': [cross_cov.transpose() for cross_cov in cross_cov_list]}
            associated_dependence_test = dependence_test_associator.associate_tracks(tracks_radar[:i + 1],
                                                                                     tracks_ais[:i + 1],
                                                                                     cross_cov_ij=cross_cov_list[
                                                                                                  :i + 1],
                                                                                     cross_cov_ji=[cross_cov.transpose()
                                                                                                   for cross_cov
                                                                                                   in cross_cov_list[
                                                                                                      :i + 1]]
                                                                                     )
            # save the associations in dicts
            num_correct_associations_counting += associated_counting
            num_false_mis_associations_counting += not associated_counting
            num_correct_associations_independent_hypothesis += associated_independence_test
            num_false_mis_associations_independent_hypothesis += not associated_independence_test
            num_correct_associations_dependent_hypothesis += associated_dependence_test
            num_false_mis_associations_dependent_hypothesis += not associated_dependence_test

        # save the number of correct associations and false mis associations in a dict
        stats_individual = {'seed': seed, 'num_correct_associations': num_correct_associations_counting,
                            'num_false_mis_associations': num_false_mis_associations_counting,
                            'type': "counting"}
        stats.append(stats_individual)
        stats_individual = {'seed': seed, 'num_correct_associations': num_correct_associations_independent_hypothesis,
                            'num_false_mis_associations': num_false_mis_associations_independent_hypothesis,
                            'type': "independent"}
        stats.append(stats_individual)
        stats_individual = {'seed': seed, 'num_correct_associations': num_correct_associations_dependent_hypothesis,
                            'num_false_mis_associations': num_false_mis_associations_dependent_hypothesis,
                            'type': "dependent"}
        stats.append(stats_individual)

        # print the percentage correct associations for each technique
        percentage_correct_counting = num_correct_associations_counting/(
                num_correct_associations_counting + num_false_mis_associations_counting
        )
        percentage_correct_independent_hypothesis = num_correct_associations_independent_hypothesis / (
                num_correct_associations_independent_hypothesis + num_false_mis_associations_independent_hypothesis
        )
        percentage_correct_dependent_hypothesis = num_correct_associations_dependent_hypothesis/(
                num_correct_associations_dependent_hypothesis + num_false_mis_associations_dependent_hypothesis
        )
        text = "counting: " + str(percentage_correct_counting) +\
               ", independent: " + str(percentage_correct_independent_hypothesis) + \
               ", dependent: " + str(percentage_correct_dependent_hypothesis)
        print(text)

# calc the #correct_associations and #false_mis_associations for each associating technique
tot_num_correct_associations_counting = sum(
    [stat['num_correct_associations'] for stat in stats if stat['type'] == "counting"]
)
tot_num_false_mis_associations_counting = sum(
    [stat['num_false_mis_associations'] for stat in stats if stat['type'] == "counting"]
)
tot_num_correct_associations_independent_hypothesis = sum(
    [stat['num_correct_associations'] for stat in stats if stat['type'] == "independent"]
)
tot_num_false_mis_associations_independent_hypothesis = sum(
    [stat['num_false_mis_associations'] for stat in stats if stat['type'] == "independent"]
)
tot_num_correct_associations_dependent_hypothesis = sum(
    [stat['num_correct_associations'] for stat in stats if stat['type'] == "dependent"]
)
tot_num_false_mis_associations_dependent_hypothesis = sum(
    [stat['num_false_mis_associations'] for stat in stats if stat['type'] == "dependent"]
)

# todo: print the total percentage correct associations
tot_percentage_correct_counting = tot_num_correct_associations_counting / (
        tot_num_correct_associations_counting + tot_num_false_mis_associations_counting
)
tot_percentage_correct_independent_hypothesis = tot_num_correct_associations_independent_hypothesis / (
        tot_num_correct_associations_independent_hypothesis + tot_num_false_mis_associations_independent_hypothesis
)
tot_percentage_correct_dependent_hypothesis = tot_num_correct_associations_dependent_hypothesis / (
        tot_num_correct_associations_dependent_hypothesis + tot_num_false_mis_associations_dependent_hypothesis
)
text = "Total: \ncounting: " + str(tot_percentage_correct_counting) + \
       ", independent: " + str(tot_percentage_correct_independent_hypothesis) + \
       ", dependent: " + str(tot_percentage_correct_dependent_hypothesis)

print("")
