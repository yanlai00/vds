import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import os.path as osp
from plot_utils import truncate_list_of_list_to_rectangular

def graph(log_dir, keys, graph=True, return_results=False, ax=None, label=None):
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
        ax.plot(results[keys[0]], results[keys[1]], label=label)
    if return_results:
        return results

def graph_multiple(log_dirs, keys, title='MazeB-v0', label=None, ax=None):
    all_results = {}
    for key in keys:
        all_results[key] = []
    for log_dir in log_dirs:
        results = graph(log_dir, keys, graph=False, return_results=True, ax=ax)
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

    sample_indices = range(0, len(X), 2)
    X = X[sample_indices]
    Y_mean = Y_mean[sample_indices]
    Y_std = Y_std[sample_indices]
    
    ax.plot(X, Y_mean, label=label)
    ax.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, alpha=.1)
    ax.set_title(title)
    ax.set_xlabel('timesteps')
    ax.set_ylabel('success rate')
    ax.set_ylim([0, 1.05])

def graph_baseline(env, title='MazeB-v0', label=None, ax=None):
    X = [25000 * i for i in range(17)]
    if env == 'MazeA-v0':
        Y_mean = [0.06, 0.17, 0.32, 0.59, 0.65, 0.75, 0.80, 0.85, 0.85, 0.93, 0.92, 0.94, 0.93, 0.94, 0.95, 0.96, 0.96]
    elif env == 'MazeB-v0':
        Y_mean = [0.07, 0.18, 0.33, 0.42, 0.52, 0.54, 0.68, 0.72, 0.78, 0.80, 0.81, 0.85, 0.91, 0.91, 0.92, 0.92, 0.94]
    elif env == "FetchPush-v1":
        X = [50000 * i for i in range(11)]
        Y_mean = [0.02, 0.08, 0.08, 0.10, 0.19, 0.29, 0.53, 0.71, 0.81, 0.86, 0.93]
    else:
        print("Unrecognized Environment!")
    ax.plot(X, Y_mean, label=label)
    ax.set_title(title)
    ax.set_xlabel('timesteps')
    ax.set_ylabel('success rate')
    ax.set_ylim([0, 1.05])
