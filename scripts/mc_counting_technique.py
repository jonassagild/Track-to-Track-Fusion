


# seeds
import numpy as np
from stonesoup.types.state import GaussianState

from trackers.kf_dependent_fusion_async_sensors import KalmanFilterDependentFusionAsyncSensors
from utils import open_object
from utils.scenario_generator import generate_scenario_3

start_seed = 0
end_seed = 150  # normally 500
num_mc_iterations = end_seed - start_seed

# params
save_fig = False

# scenario parameters
sigma_process_list = [0.3] * 11  # [0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 3, 3, 3]
sigma_meas_radar_list = [50] * 11  # [5, 30, 200, 5, 30, 200, 5, 30, 200]
sigma_meas_ais_list = [10] * 11  # [10] * 9
radar_meas_rate = 1
ais_meas_rate_list = list(range(2, 13))  # [6] * 9
timesteps = 200
num_steps_ais_ignore = 100

# dicts to store final results for printing in a latex friendly way
something_overall = {}
something_else_overall = {}

for sigma_process, sigma_meas_radar, sigma_meas_ais, ais_meas_rate in zip(sigma_process_list, sigma_meas_radar_list,
                                                           sigma_meas_ais_list, ais_meas_rate_list):
    num_steps_metrics = (timesteps - num_steps_ais_ignore) * ais_meas_rate
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
        initial_covar = np.diag([sigma_meas_radar * sigma_meas_ais, sigma_meas_radar*sigma_process,
                                 sigma_meas_radar * sigma_meas_ais, sigma_meas_radar*sigma_process]) ** 2
        prior = GaussianState([1, 1.1, -1, 0.9], initial_covar, timestamp=start_time)

        kf_dependent_fusion = KalmanFilterDependentFusionAsyncSensors(start_time, prior,
                                                                      sigma_process_radar=sigma_process,
                                                                      sigma_process_ais=sigma_process,
                                                                      sigma_meas_radar=sigma_meas_radar,
                                                                      sigma_meas_ais=sigma_meas_ais)

        tracks_fused_dependent, _, _ = kf_dependent_fusion.track_async(start_time, measurements_radar, measurements_ais,
                                                                       fusion_rate=1)

        # todo: do something with association here

        # todo count the number of associations that turn out to be correct

        # todo
