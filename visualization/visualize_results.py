import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def calc_stats(result_dict):
    means = {}
    medians = {}
    se = {}
    percentile_25 = {}
    percentile_75 = {}

    for k in result_dict.keys():
        means[k] = np.mean(result_dict[k])
        medians[k] = np.median(result_dict[k])
        se[k] = np.std(result_dict[k]) / math.sqrt(len(result_dict[k]))
        percentile_25[k] = np.percentile(result_dict[k], 25, axis=0)
        percentile_75[k] = np.percentile(result_dict[k], 75, axis=0)

    return means, medians, se, percentile_25, percentile_75


def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def collect_results(base_dir):
    # List to store the result rows
    results_list = []

    # Iterate over all directories in the base directory
    for input_dir in os.listdir(base_dir):
        input_path = os.path.join(base_dir, input_dir)
        if os.path.isdir(input_path):
            scores_file = os.path.join(input_path, 'scores.pkl')
            if os.path.exists(scores_file):
                scores_dict = read_pickle(scores_file)
                means, medians, se, percentile_25, percentile_75 = calc_stats(scores_dict)
                for key in scores_dict.keys():
                    result_row = {
                        'Input': input_dir,
                        'Key': key,
                        'Mean': means[key],
                        'Median': medians[key],
                        'SE': se[key],
                        'Percentile_25': percentile_25[key],
                        'Percentile_75': percentile_75[key]
                    }
                    results_list.append(result_row)

    # Create DataFrame from the results list
    results_df = pd.DataFrame(results_list)
    print(results_df)
    return results_df


def plot_mean_with_se(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    inputs = df['Input'].unique()

    bar_width = 0.35
    index = np.arange(len(inputs))

    mean_test = df[df['Key'] == 'test']['Mean'].values
    se_test = df[df['Key'] == 'test']['SE'].values
    mean_val = df[df['Key'] == 'val']['Mean'].values
    se_val = df[df['Key'] == 'val']['SE'].values

    ax.bar(index, mean_test, bar_width, yerr=se_test, label='Test', alpha=0.7, capsize=5, zorder=2)
    ax.bar(index + bar_width, mean_val, bar_width, yerr=se_val, label='Validation', alpha=0.7, capsize=5, zorder=2)

    ax.set_xlabel('Input')
    ax.set_ylabel('Mean Performance')
    ax.set_title('Mean Performance with Standard Error Bars')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(inputs, rotation=45)
    ax.legend()

    # Set grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(which='major', axis='y', linestyle='-', linewidth='0.75', color='0.93', zorder=0)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth='0.5', color='0.95', zorder=0)

    plt.tight_layout()
    plt.show()


def plot_median_with_iqr(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    inputs = df['Input'].unique()

    bar_width = 0.35
    index = np.arange(len(inputs))

    median_test = df[df['Key'] == 'test']['Median'].values
    p25_test = df[df['Key'] == 'test']['Percentile_25'].values
    p75_test = df[df['Key'] == 'test']['Percentile_75'].values

    median_val = df[df['Key'] == 'val']['Median'].values
    p25_val = df[df['Key'] == 'val']['Percentile_25'].values
    p75_val = df[df['Key'] == 'val']['Percentile_75'].values

    ax.bar(index, median_test, bar_width, yerr=[median_test - p25_test, p75_test - median_test], label='Test',
           alpha=0.7, capsize=5, zorder=2)
    ax.bar(index + bar_width, median_val, bar_width, yerr=[median_val - p25_val, p75_val - median_val],
           label='Validation', alpha=0.7, capsize=5, zorder=2)

    ax.set_xlabel('Input')
    ax.set_ylabel('Median Performance')
    ax.set_title('Median Performance with IQR Error Bars')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(inputs, rotation=45)
    ax.legend()

    # Set grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(which='major', axis='y', linestyle='-', linewidth='0.75', color='0.93', zorder=0)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth='0.5', color='0.95', zorder=0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root_path = "/Volumes/T5 EVO/Overfitting/out/loss"
    results = collect_results(root_path)
    plot_mean_with_se(results)
    plot_median_with_iqr(results)
