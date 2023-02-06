from pathlib import Path
import pickle
import numpy as np

def per_algo_per_noise_tables(data):
    noise_scenarios = set([row[2] for row in data])
    tables = {}
    for noise_scenario in noise_scenarios:
        #Data with only rows with noise_scenario
        sub_data = [row for row in data if row[2] == noise_scenario]
        sub_table = per_algo_table(sub_data)
        tables[noise_scenario] = sub_table
    return tables

def per_algo_table(data):
    result_table = {}
    algos = set([row[1] for row in data])
    #Make one row per algorithm with the key being the name of the algorithm
    for algo in algos:
        result_table[algo] = np.zeros((10,))
        #Compute average error for each algorithm
        nb_rows = 0
        for row in data:
            object_name, a, n, geodesic_error, size_based_error, relative_error, rmse_error = row
            if a == algo:
                nb_rows += 1
                result_table[algo][1:4] += np.array([size_based_error[0], size_based_error[1],size_based_error[2]])
                if geodesic_error != -1:
                    result_table[algo][0] += 100
                    result_table[algo][7] += geodesic_error
                else:
                    result_table[algo][7] += np.NaN
                for i in range(3):
                    if relative_error[i] != np.Inf:
                        result_table[algo][4+i] += relative_error[i]
                result_table[algo][8] += np.array(np.mean(rmse_error[0:3]))
                result_table[algo][9] += np.array(np.mean(rmse_error[3:6]))
        result_table[algo] /= nb_rows
    return result_table

EXPERIMENTS_BASE_PATH = Path('./Experiments')
DATA_FILE = EXPERIMENTS_BASE_PATH / Path('results.pkl')

if DATA_FILE.exists() and not DATA_FILE.is_dir():
    path = DATA_FILE.resolve().as_posix()
    with open(path, "rb") as input_file:
        data = pickle.load(input_file)
        #Noise scenarios
        tables = per_algo_per_noise_tables(data)
        for t in ['No','Low','Mod','High']:
            try:
                table = tables[t]
                print('{} noise'.format(t))
                print('Algo. \t Cons. \t Rie \t Mass \t Com \t Inertia')
                for k in table.keys():
                    print('{} \t {} \t {} \t {} \t {} \t {}'.format(k, np.round(table[k][0],2), np.round(table[k][7],2), np.round(table[k][1],2), np.round(table[k][2],2), np.round(table[k][3],2)))
            except:
                pass
