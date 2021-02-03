"""script for plotting and testing association techniques

The script uses the Kalman Filter from Stone Soup to produce tracks in a similar fashion as done in the Tracker Classes.


"""
import numpy as np
import scipy
from stonesoup.types.state import GaussianState
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from data_association.CountingAssociator import CountingAssociator
from data_association.bar_shalom_hypothesis_associators import HypothesisTestIndependenceAssociator, \
    HypothesisTestDependenceAssociator
from trackers.kf_dependent_fusion_async_sensors import KalmanFilterDependentFusionAsyncSensors

from utils.scenario_generator import generate_scenario_3
from utils import open_object
from utils.save_figures import save_figure

save_fig = False

seed = 1996
radar_meas_rate = 1
ais_meas_rate = 5
timesteps = 4  # num of measurements from the slowest sensor
sigma_process = 1
sigma_meas_radar = 5
sigma_meas_ais = 10

num_estimates = timesteps + max(ais_meas_rate, radar_meas_rate)

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
prior = GaussianState([0, 1, 0, 1], np.diag([1.5, 0.5, 1.5, 0.5]) ** 2, timestamp=start_time)

kf_dependent_fusion = KalmanFilterDependentFusionAsyncSensors(start_time, prior,
                                                              sigma_process_radar=sigma_process,
                                                              sigma_process_ais=sigma_process,
                                                              sigma_meas_radar=sigma_meas_radar,
                                                              sigma_meas_ais=sigma_meas_ais)

# hacky way; just so its easy to reuse code
measurement_model_radar = kf_dependent_fusion.measurement_model_radar
measurement_model_ais = measurement_model_radar

tracks_fused, tracks_radar, tracks_ais = kf_dependent_fusion.track_async(start_time,
                                                                         measurements_radar,
                                                                         measurements_ais,
                                                                         fusion_rate=1)
cross_cov_list = kf_dependent_fusion.cross_cov_list
# counting technique parameters
association_distance_threshold = 10
consecutive_hits_confirm_association = 3
consecutive_misses_end_association = 2
counting_associator = CountingAssociator(association_distance_threshold, consecutive_hits_confirm_association,
                                         consecutive_misses_end_association)
independence_test_associator = HypothesisTestIndependenceAssociator()
dependence_test_associator = HypothesisTestDependenceAssociator()

# plot and check for association, one by one
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.axis('equal')
ax.plot([state.state_vector[0] for state in ground_truth],
        [state.state_vector[2] for state in ground_truth],
        linestyle="--",
        label='Ground truth')
ax.scatter([state.state_vector[0] for state in measurements_radar],
           [state.state_vector[1] for state in measurements_radar],
           color='b',
           label='Measurements Radar')
ax.scatter([state.state_vector[0] for state in measurements_ais],
           [state.state_vector[1] for state in measurements_ais],
           color='r',
           label='Measurements AIS')

# add ellipses to add legend
for text, color in zip(['Posterior AIS', 'Posterior Radar'], ['r', 'b']):
    ellipse = Ellipse(xy=(0, 0),
                      width=0,
                      height=0,
                      color=color,
                      alpha=0.2,
                      label=text)
    ax.add_patch(ellipse)

# add ellipses to the posteriors
for i in range(0, len(tracks_radar)):
    state_radar = tracks_radar[i]
    state_ais = tracks_ais[i]

    # add ellipse to figure
    for state, color in zip([state_radar, state_ais], ['b', 'r']):
        w, v = np.linalg.eig(measurement_model_radar.matrix() @ state.covar @ measurement_model_radar.matrix().T)
        max_ind = np.argmax(w)
        min_ind = np.argmin(w)
        orient = np.arctan2(v[1, max_ind], v[0, max_ind])
        ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
                          width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
                          angle=np.rad2deg(orient),
                          alpha=0.2,
                          color=color)
        ax.add_artist(ellipse)

    # todo: check association, print association and display figure
    associated_counting_technique = counting_associator.associate_tracks(tracks_radar[:i + 1], tracks_ais[:i + 1])
    associated_independence_test = independence_test_associator.associate_tracks(tracks_radar[:i + 1],
                                                                                 tracks_ais[:i + 1])
    cross_cov_dict = {'cross_cov_ij': cross_cov_list,
                      'cross_cov_ji': [cross_cov.transpose() for cross_cov in cross_cov_list]}
    associated_dependence_test = dependence_test_associator.associate_tracks(tracks_radar[:i + 1],
                                                                             tracks_ais[:i + 1],
                                                                             cross_cov_ij=cross_cov_list[:i + 1],
                                                                             cross_cov_ji=[cross_cov.transpose()
                                                                                           for cross_cov
                                                                                           in cross_cov_list[:i + 1]]
                                                                             )

    ax.legend(prop={'size': 12})
    title = "counting association: " + associated_counting_technique.__str__() + \
            ",\n independent hypothesis association: " + associated_independence_test.__str__() + \
            ",\n dependent hypothesis association: " + associated_dependence_test.__str__()
    ax.set_title(title, fontsize=20)
    fig.show()
    # todo: add additional information, as num consecutive hits and misses, within threshold etc
    print("next")

ax.legend(prop={'size': 12})
title = "Figure for testing association"
ax.set_title(title, fontsize=20)
fig.show()
if save_fig:
    folder = "../results/final_results/scenario_examples"
    name = "scenario2_example"
    save_figure(folder, name + ".pdf", fig)

# # plot estimate for estimate
# # plot
# fig_2 = plt.figure(figsize=(10, 6))
# ax = fig_2.add_subplot(1, 1, 1)
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
# ax.axis('equal')
# ax.plot([state.state_vector[0] for state in ground_truth],
#         [state.state_vector[2] for state in ground_truth],
#         linestyle="--",
#         label='Ground truth')
# # ax.scatter([state.state_vector[0] for state in measurements_radar],
# #            [state.state_vector[1] for state in measurements_radar],
# #            color='b',
# #            label='Measurements Radar')
# # ax.scatter([state.state_vector[0] for state in measurements_ais],
# #            [state.state_vector[1] for state in measurements_ais],
# #            color='r',
# #            label='Measurements AIS')
#
# for i in range(0, len(tracks_fused)):
#     # plot measurements
#     ax.scatter([measurements_radar[i + 1].state_vector[0]],
#                [measurements_radar[i + 1].state_vector[1]],
#                color='b',
#                label='Measurements Radar')
#     ax.scatter([measurements_ais[i + 1].state_vector[0]],
#                [measurements_ais[i + 1].state_vector[1]],
#                color='r',
#                label='Measurements AIS')
#
#     # plot one and one estimate
#     state_radar = tracks_radar[i + 1]
#     w, v = np.linalg.eig(measurement_model_radar.matrix() @ state_radar.covar @ measurement_model_radar.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state_radar.state_vector[0], state_radar.state_vector[2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.2,
#                       color='b')
#     ax.add_artist(ellipse)
#
#     state_ais = tracks_ais[i + 1]
#     w, v = np.linalg.eig(measurement_model_ais.matrix() @ state_ais.covar @ measurement_model_ais.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state_ais.state_vector[0], state_ais.state_vector[2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.2,
#                       color='r')
#     ax.add_patch(ellipse)
#
#     state_fused = tracks_fused[i]
#     w, v = np.linalg.eig(measurement_model_ais.matrix() @ state_fused.covar @ measurement_model_ais.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state_fused.state_vector[0], state_fused.state_vector[2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.5,
#                       color='green')
#     ax.add_patch(ellipse)
#
#     fig_2.show()
#     input("Press Enter to continue...")


#
# # add ellipses to the posteriors
# for state in tracks_radar:
#     w, v = np.linalg.eig(measurement_model_radar.matrix() @ state.covar @ measurement_model_radar.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.2,
#                       color='b')
#     ax.add_artist(ellipse)
#
# for state in tracks_ais:
#     w, v = np.linalg.eig(measurement_model_ais.matrix() @ state.covar @ measurement_model_ais.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.2,
#                       color='r')
#     ax.add_patch(ellipse)
#
# for track_fused in tracks_fused:
#     w, v = np.linalg.eig(measurement_model_ais.matrix() @ track_fused[1] @ measurement_model_ais.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(track_fused[0][0], track_fused[0][2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.5,
#                       color='green')
#     ax.add_patch(ellipse)

# fig_2.show()
