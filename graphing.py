import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import os.path as osp
from plot_utils import truncate_list_of_list_to_rectangular

def graph(log_dir, keys, graph=True, return_results=False):
    results = {}
    for key in keys:
        results[key] = []  
    with open(osp.join(log_dir, 'progress.csv'), newline='') as csvfile:
        results_reader = csv.DictReader(csvfile)
        for row in results_reader:
            for key in keys:
                results[key].append(row[key])
    for key in keys:
        results[key] = np.array(results[key]).astype(np.float)
    if graph:
        plt.plot(results[keys[0]], results[keys[1]])
        plt.show()
    if return_results:
        return results

def graph_multiple(log_dirs, keys, title='MazeB-v0'):
    all_results = {}
    for key in keys:
        all_results[key] = []
    for log_dir in log_dirs:
        results = graph(log_dir, keys, graph=False, return_results=True)
        all_results[keys[0]] = results[keys[0]]
        for key in keys[1:]:
            all_results[key].append(results[key])
    for key in keys[1:]:
        all_results[key], min_length = truncate_list_of_list_to_rectangular(all_results[key])
        all_results[key] = np.array(all_results[key]).astype(np.float)
        all_results[key + '_mean'] = np.mean(all_results[key], axis=0)
        all_results[key + '_std'] = np.std(all_results[key], axis=0)
    all_results[keys[0]] = all_results[keys[0]][:min_length]
    all_results[keys[0]] = np.array(all_results[keys[0]]).astype(np.float)
    
    X = all_results[keys[0]]
    Y_mean = all_results[keys[1] + '_mean']
    Y_std = all_results[keys[1] + '_std']
    
    plt.plot(X, Y_mean)
    plt.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, alpha=.1)
    plt.title(title)
    plt.xlabel('timesteps')
    plt.ylabel('success rate')
    plt.ylim([0, 1.05])
    plt.show()
