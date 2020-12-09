"""monte_carlo_simulations Script for running monte carlo simulations

Script used for running monte carlo simulations.
"""
import numpy as np
import scipy
from stonesoup.types.state import GaussianState
from matplotlib import pyplot as plt

from utils import open_object
from utils import calc_metrics

from utils.scenario_generator import generate_scenario_3
from trackers.kf_kf_fusion_unsync_sensors import KFFusionUnsyncSensors
from trackers.kf_independent_fusion_unsync_sensors import kalman_filter_independent_fusion
from trackers.kalman_filter_dependent_fusion import kalman_filter_dependent_fusion

from utils.save_figures import save_figure

# seeds
start_seed = 0
end_seed = 5  # normally 500
num_mc_iterations = end_seed - start_seed

# params
save_fig = False

# scenario parameters
sigma_process_list = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 3, 3, 3]
sigma_meas_radar_list = [1, 5, 100, 1, 5, 100, 1, 5, 100]
sigma_meas_ais_list = [10, 10, 10, 10, 10, 10, 10, 10, 10]
radar_meas_rate = 1
ais_meas_rate = 5
timesteps = 10

for sigma_process, sigma_meas_radar, sigma_meas_ais in zip(sigma_process_list, sigma_meas_radar_list,
                                                           sigma_meas_ais_list):
    stats_overall = []
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
        prior = GaussianState([0, 1, 0, 1], np.diag([1.5, 0.5, 1.5, 0.5]) ** 2, timestamp=start_time)

        # trackers
        kf_kf_fusion = KFFusionUnsyncSensors(start_time,
                                             prior,
                                             sigma_process=sigma_process,
                                             sigma_meas_radar=sigma_meas_radar,
                                             sigma_meas_ais=sigma_meas_ais)
        kf_independent_fusion = kalman_filter_independent_fusion(start_time,
                                                                 prior,
                                                                 sigma_process_radar=sigma_process,
                                                                 sigma_process_ais=sigma_process,
                                                                 sigma_meas_radar=sigma_meas_radar,
                                                                 sigma_meas_ais=sigma_meas_ais)
        # kf_dependent_fusion = kalman_filter_dependent_fusion(measurements_radar, measurements_ais, start_time, prior,
        #                                                      sigma_process_radar=sigma_process,
        #                                                      sigma_process_ais=sigma_process,
        #                                                      sigma_meas_radar=sigma_meas_radar,
        #                                                      sigma_meas_ais=sigma_meas_ais)

        tracks_fused_independent, _, _ = kf_independent_fusion.track(start_time, measurements_radar, measurements_ais,
                                                                     fusion_rate=1)
        # tracks_fused_dependent, _, _ = kf_dependent_fusion.track()
        tracks_kf_fusion, _ = kf_kf_fusion.track(measurements_radar, measurements_ais, estimation_rate=1)
        #
        # # fix the length of the fusions to dependent (as it is a bit shorter)
        num_tracks = len(tracks_fused_independent)
        # # get the num_tracks last elements
        # tracks_fused_independent = tracks_fused_independent[-num_tracks:]
        tracks_kf_fusion = tracks_kf_fusion[-num_tracks:]
        # remove the first element of ground_truth (because we only fuse the n-1 last with dependent fusion)
        ground_truth = ground_truth[-num_tracks:]

        # Calculate some metrics
        stats_individual = {}
        # calculate NEES
        stats_individual["NEES"] = calc_metrics.calc_nees(tracks_fused_independent, ground_truth)
        # calculate ANEES
        stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])
        # calculate RMSE
        stats_individual["RMSE"] = calc_metrics.calc_rmse(tracks_fused_independent, ground_truth)
        stats_individual["seed"] = seed
        stats_individual["fusion_type"] = "independent"
        stats_overall.append(stats_individual)
        #
        # stats_individual = {}
        # # calculate NEES
        # stats_individual["NEES"] = calc_metrics.calc_nees(tracks_fused_dependent, ground_truth)
        # # calculate ANEES
        # stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])
        # # calculate RMSE
        # stats_individual["RMSE"] = calc_metrics.calc_rmse(tracks_fused_dependent, ground_truth)
        # stats_individual["seed"] = seed
        # stats_individual["fusion_type"] = "dependent"
        # stats_overall.append(stats_individual)
        #
        stats_individual = {}
        # calculate NEES
        stats_individual["NEES"] = calc_metrics.calc_nees(tracks_kf_fusion, ground_truth)
        # calculate ANEES
        stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])
        # calculate RMSE
        stats_individual["RMSE"] = calc_metrics.calc_rmse(tracks_kf_fusion, ground_truth)
        stats_individual["seed"] = seed
        stats_individual["fusion_type"] = "ais as measurement"
        stats_overall.append(stats_individual)

    # plot some results
    alpha = 0.95
    ci_nees = scipy.stats.chi2.interval(alpha, 4)
    ci_anees = np.array(scipy.stats.chi2.interval(alpha, 4 * num_tracks)) / num_tracks

    # plot ANEES and confidence interval
    fig_ci_anees = plt.figure(figsize=(10, 6))
    ax_ci_anees = fig_ci_anees.add_subplot(1, 1, 1)
    ax_ci_anees.set_xlabel("MC seed")
    ax_ci_anees.set_ylabel("ANEES")

    # plot upper and lower confidence intervals
    ax_ci_anees.plot([start_seed, end_seed], [ci_anees[0], ci_anees[0]], color='red')
    ax_ci_anees.plot([start_seed, end_seed], [ci_anees[1], ci_anees[1]], color='red')

    anees_independent = [stat['ANEES'] for stat in stats_overall if stat["fusion_type"] == "independent"]
    # anees_dependent = [stat['ANEES'] for stat in stats_overall if stat["fusion_type"] == "dependent"]
    anees_ais_as_measurement = [stat['ANEES'] for stat in stats_overall if stat["fusion_type"] == "ais as measurement"]

    rmse_independent = np.mean([stat['RMSE'] for stat in stats_overall if stat["fusion_type"] == "independent"])
    # rmse_dependent = np.mean([stat['RMSE'] for stat in stats_overall if stat["fusion_type"] == "dependent"])
    rmse_ais_as_measurement = np.mean([stat['RMSE'] for stat in stats_overall if stat["fusion_type"] == "ais as "
                                                                                                        "measurement"])

    # plot the ANEES values
    ax_ci_anees.plot(list(range(start_seed, end_seed)), anees_independent, marker='+', ls='None', color='blue',
                     label='Independent')
    # ax_ci_anees.plot(list(range(start_seed, end_seed)), anees_dependent, marker='+', ls='None', color='red',
    #                  label='Dependent')
    ax_ci_anees.plot(list(range(start_seed, end_seed)), anees_ais_as_measurement, marker='+', ls='None', color='green',
                     label='AIS as measurement')

    ax_ci_anees.legend()
    # fig_ci_anees.show()

    # save_figure("../results/scenario2/1996", "monte_carlo_all_trackers_same_params.svg", fig_ci_anees)

    # plot to show how the average ANEES approaches some limit. Plot a changing confidence interval?
    fig_ci_average_anees = plt.figure(figsize=(10, 6))
    ax_ci_average_anees = fig_ci_average_anees.add_subplot(1, 1, 1)
    ax_ci_average_anees.set_xlabel("MC iterations")
    ax_ci_average_anees.set_ylabel("ANEES")
    # set y-axis limits
    ax_ci_average_anees.set_ylim(2.5, 27)

    ci_average_anees = [
        np.array(scipy.stats.chi2.interval(alpha, 4 * num_tracks * num_MC_it)) / (num_tracks * num_MC_it)
        for num_MC_it in range(1, end_seed - start_seed + 1)]
    ci_average_anees_lower = [ci[0] for ci in ci_average_anees]
    ci_average_anees_upper = [ci[1] for ci in ci_average_anees]

    # not the quickest, but works
    average_anees_independent_list = [np.average(anees_independent[:idx]) for idx in range(1, num_mc_iterations + 1)]
    # average_anees_dependent_list = [np.average(anees_dependent[:idx]) for idx in range(1, num_mc_iterations + 1)]
    average_anees_ais_as_measurement_list = [np.average(anees_ais_as_measurement[:idx])
                                             for idx in range(1, num_mc_iterations + 1)]

    # plot upper and lower confidence intervals
    ax_ci_average_anees.plot(list(range(start_seed, end_seed)), ci_average_anees_lower, color='black',
                             label='Confidence Intervals')
    ax_ci_average_anees.plot(list(range(start_seed, end_seed)), ci_average_anees_upper, color='black')

    # plot the average ANEES
    ax_ci_average_anees.plot(list(range(start_seed, end_seed)), average_anees_independent_list, color='blue',
                             label='Independent')
    # ax_ci_average_anees.plot(list(range(start_seed, end_seed)), average_anees_dependent_list, color='red',
    #                          label='Dependent')
    ax_ci_average_anees.plot(list(range(start_seed, end_seed)), average_anees_ais_as_measurement_list, color='green',
                             label='AIS as measurement')

    # set title
    title = "Average ANEES with parameters $\sigma_{process}=$" + str(sigma_process) + ",  $\sigma_{AIS}=$" + \
            str(sigma_meas_ais) + ", $\sigma_{Radar}=$" + str(sigma_meas_radar)
    ax_ci_average_anees.set_title(title)

    ax_ci_average_anees.legend()
    fig_ci_average_anees.show()

    if save_fig:
        fig_name = "process_" + str(sigma_process) + "_AIS_" + str(sigma_meas_ais) + "_Radar_" \
                   + str(sigma_meas_radar) + ".pdf"
        save_figure("../results/scenario3/mc_average_anees", fig_name, fig_ci_average_anees)

    # print some results
    print("Process: " + str(sigma_process) + " AIS: " + str(sigma_meas_ais) + " Radar: " + str(sigma_meas_radar))
    print("Average ANEES independent: " + str(average_anees_independent_list[-1]))
    # print("Average ANEES dependent: " + str(average_anees_dependent_list[-1]))
    print("Average ANEES AIS as measurement: " + str(average_anees_ais_as_measurement_list[-1]))
    print("RMSE independent: " + str(rmse_independent))
    # print("RMSE dependent: " + str(rmse_dependent))
    print("RMSE AIS as measurement: " + str(rmse_ais_as_measurement))
    print("")

print("CI intervals: " + str(ci_average_anees_lower[-1]) + ", " + str(ci_average_anees_upper[-1]))
