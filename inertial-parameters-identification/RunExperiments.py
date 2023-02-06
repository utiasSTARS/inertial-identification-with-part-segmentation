from time import time_ns
from InertialParametersEstimation import Estimator
from pathlib import Path
import pickle

#World name used in the with-respect-to database
world_name = 'trajsim'

#Algorithms to use for experiments
algorithms_to_test = ['HPS', 'OLS', 'GEO']

#Noise scenarios to use for experiments
noise_scenarios = ['No', 'Low', 'Mod', 'High']

#Definition of the noise scenarios
vel_noise_std_dev       = {'No': [0,0], 'Low': [0,     0],      'Mod': [0,    0],     'High': [0,   0]}
acc_noise_std_dev       = {'No': [0,0], 'Low': [0.025, 0.25],   'Mod': [0.05, 0.5],   'High': [0.1, 1]}
wrench_noise_std_dev    = {'No': [0,0], 'Low': [0.05,  0.0025], 'Mod': [0.1,  0.005], 'High': [0.2, 0.01]}

#Paths
EXPERIMENTS_BASE_PATH   = Path('./Experiments')
SEGMENTATIONS_BASE_PATH = Path("../data/Segmentations")
SIMULATIONS_BASE_PATH   = Path("../data/Simulations")
DATASET_BASE_PATH       = Path("../data/Workshop Tools Dataset")
yaml_files = DATASET_BASE_PATH.glob('*/Inertia.yaml')

avg_cond_number = {'No': 0, 'Low': 0, 'Mod': 0, 'High': 0}

test_data = []
nb_tests = len([f for f in DATASET_BASE_PATH.glob('*/Inertia.yaml')])*len(algorithms_to_test)*len(noise_scenarios)
i = 0
for yaml_path in yaml_files:
    object_name = yaml_path.parent.stem.replace(' ','_').replace('+','_')

    pkl_path  = SIMULATIONS_BASE_PATH / Path(object_name+'.pkl')
    ply_path  = SEGMENTATIONS_BASE_PATH / Path(object_name) / Path(object_name+'.ply')

    for algo in algorithms_to_test:
        
        for noise_scenario in noise_scenarios:
            v_noise_sd = vel_noise_std_dev[noise_scenario]
            a_noise_sd = acc_noise_std_dev[noise_scenario]
            ft_noise_sd= wrench_noise_std_dev[noise_scenario]
            i += 1
            print('({}/{}) Running {} with {} and {} noise'.format(i, nb_tests, object_name, algo, noise_scenario))
            est = Estimator(world_name, pkl_path, yaml_path, ply_path, algo)
            est.process_data(v_noise_sd, a_noise_sd, ft_noise_sd)
            estimate = est.estimate()
            if algo == 'OLS':
                cond_number = est.get_condition_number()
                avg_cond_number[noise_scenario] += cond_number/(nb_tests/len(noise_scenarios))
            geodesic_error   = est.evaluate(estimate, metric='geodesic')
            size_based_error = est.evaluate(estimate, metric='size-based')
            relative_error   = est.evaluate(estimate, metric='relative')
            rmse_error       = est.evaluate(estimate, metric='rmse')

            row = [object_name, algo, noise_scenario, geodesic_error, size_based_error, relative_error, rmse_error]
            test_data.append(row)

for noise_scenario in noise_scenarios:
    print('Average condition number for {} noise: {}'.format(noise_scenario, int(avg_cond_number[noise_scenario])))

#Save experiments data
output_path = EXPERIMENTS_BASE_PATH / Path('results.pkl')
output_path.parent.mkdir(parents=True, exist_ok=True)
path = output_path.resolve().as_posix()
with open(path, "wb") as output_file:
    pickle.dump(test_data, output_file, pickle.HIGHEST_PROTOCOL)
    output_file.close()
