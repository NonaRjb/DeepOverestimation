import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import re


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
    return results_df


def process_df(df):
    # Define regular expressions for each format
    pattern_dd_r = r'^\d+d_\d+\.\d+$'  # Matches Dd_r where D is an integer and r is a float
    pattern_dd_hh = r'^\d+d_\d+h$'  # Matches Dd_Hh where D and H are both integers
    pattern_hh_r = r'^\d+h_\d+\.\d+$'  # Matches Hh_r where H is an integer and r is a float
    pattern_dd_hh_r = r'^\d+d_\d+h_\d+\.\d+$' # Matches Dd_Hh_r where D and H are both integers and r is a float
    included_vars = None

    sample_input = df['Input'].iloc[0]  # Get the first 'Input' value to check the pattern

    if re.match(pattern_dd_r, sample_input):
        print("Detected format: Dd_r")
        # Extract D and r
        df['d'] = df['Input'].str.extract(r'(\d+)d')[0].astype(int)
        df['r'] = df['Input'].str.extract(r'd_(\d+\.\d+)')[0].astype(float)
        included_vars = ['d', 'r']

    elif re.match(pattern_dd_hh, sample_input):
        print("Detected format: Dd_Hh")
        # Extract D and H
        df['d'] = df['Input'].str.extract(r'(\d+)d')[0].astype(int)
        df['h'] = df['Input'].str.extract(r'd_(\d+)h')[0].astype(int)
        included_vars = ['d', 'h']

    elif re.match(pattern_hh_r, sample_input):
        print("Detected format: Hh_r")
        # Extract H and r
        df['h'] = df['Input'].str.extract(r'(\d+)h')[0].astype(int)
        df['r'] = df['Input'].str.extract(r'h_(\d+\.\d+)')[0].astype(float)
        included_vars = ['h', 'r']

    elif re.match(pattern_dd_hh_r, sample_input):
        print("Detected format: Dd_Hh_r")
        # Extract D and H and r
        df['d'] = df['Input'].str.extract(r'(\d+)d')[0].astype(int)
        df['h'] = df['Input'].str.extract(r'd_(\d+)h')[0].astype(int)
        df['r'] = df['Input'].str.extract(r'h_(\d+\.\d+)')[0].astype(float)
        included_vars = ['d', 'h', 'r']

    else:
        print("Unknown format")

    return df, included_vars


def plot_across_var(df, x, y=None, save_path=None):
    df_new, included_vars = process_df(df)

    # Get the unique values of the variable x
    unique_var_values = df_new[x].unique()

    # Get the second varying variable that isn't x or y
    if y:
        second_var = next((item for item in included_vars if item != x and item != y), None)
    else:
        second_var = next((item for item in included_vars if item != x), None)

    for v in unique_var_values:
        # Filter the dataframe by x = v
        df_v = df_new[df_new[x] == v]

        if y:
            # Loop through unique values of y and plot for each combination of x and y
            unique_y_values = df_v[y].unique()
            for y_val in unique_y_values:
                df_y = df_v[df_v[y] == y_val]
                if second_var is not None:
                    df_y = df_y.sort_values(by=second_var)
                fig, ax = plt.subplots(1, 2, figsize=(16, 6))
                plot_mean_with_se(df_y, ax=ax[0])
                plot_median_with_iqr(df_y, ax=ax[1])
                fig.suptitle(f"{x} = {v}, {y} = {y_val}", fontsize=18, y=0.97)
                plt.subplots_adjust(top=0.9)
                if save_path:
                    plt.savefig(os.path.join(save_path, f"{x}{v}_{y}{y_val}.png"))
                plt.close()
        else:
            # Sort by the second variable if it exists
            if second_var is not None:
                df_v = df_v.sort_values(by=second_var)
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            plot_mean_with_se(df_v, ax=ax[0])
            plot_median_with_iqr(df_v, ax=ax[1])
            if x == 'd':
                fig.suptitle(f"data dimensionality = {v}", fontsize=18, y=0.97)
            elif x == 'r':
                fig.suptitle(f"ratio of large eigen values = {v}", fontsize=18, y=0.97)
            elif x == 'h':
                fig.suptitle(f"hidden size = {v}", fontsize=18, y=0.97)
            plt.subplots_adjust(top=0.9)
            if save_path:
                plt.savefig(os.path.join(save_path, f"{x}{v}.png"))
            plt.close()


def plot_mean_with_se(df, ax=None, save_path=None, figname="test.png"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    inputs = df['Input'].unique()

    bar_width = 0.35
    index = np.arange(len(inputs))

    mean_test = df[df['Key'] == 'test']['Mean'].values
    se_test = df[df['Key'] == 'test']['SE'].values
    mean_val = df[df['Key'] == 'val']['Mean'].values
    se_val = df[df['Key'] == 'val']['SE'].values

    # Set grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(which='major', axis='y', linestyle='-', linewidth='0.75', color='0.93', zorder=0)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth='0.75', color='0.93', zorder=0)

    ax.bar(index, mean_test, bar_width, yerr=se_test, label='Test', alpha=0.7, capsize=5, zorder=2)
    ax.bar(index + bar_width, mean_val, bar_width, yerr=se_val, label='Validation', alpha=0.7, capsize=5, zorder=2)

    ax.set_xlabel('Input')
    ax.set_ylabel('Mean Performance')
    ax.set_title('Mean Performance with Standard Error Bars')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(inputs, rotation=45)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, figname))


def plot_median_with_iqr(df, ax=None, save_path=None, figname="test.png"):
    if ax is None:
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

    # Set grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(which='major', axis='y', linestyle='-', linewidth='0.75', color='0.93', zorder=0)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth='0.75', color='0.93', zorder=0)

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

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, figname))


if __name__ == "__main__":
    experiment = "n5000_dhr"
    root_path = "/Volumes/T5 EVO/Overfitting/out/auc/"
    data_dir = os.path.join(root_path, experiment)
    save_dir = os.path.join(root_path, experiment+"_plots")
    results = collect_results(data_dir)
    plot_across_var(results, x='h', y='r', save_path=save_dir)
