import numpy as np
import pandas as pd
import math
import pickle
import os
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
                scores_dict['val-test'] = scores_dict['val'] - scores_dict['test']
                if 'val_loss' in scores_dict.keys():
                    del scores_dict['val_loss']
                if 'test_loss' in scores_dict.keys():
                    del scores_dict['test_loss']
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
    pattern_dd_hh_r = r'^\d+d_\d+h_\d+\.\d+$'  # Matches Dd_Hh_r where D and H are both integers and r is a float
    pattern_dd_hh_ll = r'^\d+d_\d+h_\d+l$'
    pattern_dd_nn_ll = r'^\d+d_\d+n_\d+l$'
    pattern_oo_hh_ll = r'^(.*?)o_(\d+)h_(\d+)l$'
    pattern_nn_oo_mm = r'^N\d+_([A-Za-z0-9]+)_([A-Za-z0-9]+)$'
    pattern_mod_ss_model = r'^real_([A-Za-z]+)_s(\d+)_([A-Za-z0-9]+)$'
    pattern_n_ch_f_model = r'^n(\d+)_ch(\d+)_f(\d+)_([A-Za-z0-9]+)$'
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

    elif re.match(pattern_dd_hh_ll, sample_input):
        print("Detected format: Dd_Hh_Ll")
        # Extract D and H and r
        df['d'] = df['Input'].str.extract(r'(\d+)d')[0].astype(int)
        df['h'] = df['Input'].str.extract(r'd_(\d+)h')[0].astype(int)
        df['l'] = df['Input'].str.extract(r'h_(\d+)l')[0].astype(int)
        included_vars = ['d', 'h', 'l']

    elif re.match(pattern_oo_hh_ll, sample_input):
        print("Detected format: Oo_Hh_Ll")
        # Extract D and H and r
        df['o'] = df['Input'].str.extract(r'(.*?)o')[0].astype(str)
        df['h'] = df['Input'].str.extract(r'o_(\d+)h')[0].astype(int)
        df['l'] = df['Input'].str.extract(r'h_(\d+)l')[0].astype(int)
        included_vars = ['o', 'h', 'l']

    elif re.match(pattern_dd_nn_ll, sample_input):
        print("Detected format: Dd_Nn_Ll")
        # Extract D and H and r
        df['d'] = df['Input'].str.extract(r'(\d+)d')[0].astype(int)
        df['n'] = df['Input'].str.extract(r'd_(\d+)n')[0].astype(int)
        df['l'] = df['Input'].str.extract(r'n_(\d+)l')[0].astype(int)
        included_vars = ['d', 'n', 'l']
    elif re.match(pattern_nn_oo_mm, sample_input):
        print("Detected format: Nn_m_o")
        df['n'] = df['Input'].str.extract(r'^N(\d+)_')[0].astype(int)
        df['m'] = df['Input'].str.extract(r'^N\d+_([A-Za-z0-9]+)_')[0]
        df['o'] = df['Input'].str.extract(r'^N\d+_[A-Za-z0-9]+_([A-Za-z0-9]+)$')[0]
        included_vars = ['n', 'm', 'o']
    elif re.match(pattern_mod_ss_model, sample_input):
        print("Detected format: mod_ss_model")
        df['mod'] = df['Input'].str.extract(r'^real_([A-Za-z]+)_')[0]
        df['s'] = df['Input'].str.extract(r'_s(\d+)_')[0].astype(int)
        df['m'] = df['Input'].str.extract(r'_s\d+_([A-Za-z0-9]+)$')[0]
        included_vars = ['mod', 's', 'm']
    elif re.match(pattern_n_ch_f_model, df['Input'][0]):  # Assuming the pattern is the same for all rows
        print("Detected format: n_ch_f_model")
        df['n'] = df['Input'].str.extract(r'^n(\d+)_')[0].astype(int)
        df['ch'] = df['Input'].str.extract(r'_ch(\d+)_')[0].astype(int)
        df['f'] = df['Input'].str.extract(r'_f(\d+)_')[0].astype(int)
        df['m'] = df['Input'].str.extract(r'_f\d+_([A-Za-z0-9]+)$')[0]
        included_vars = ['n', 'ch', 'f', 'm']

    else:
        print("Unknown format")

    return df, included_vars


def compute_diff(df, indices):
    df_pivot = df.pivot_table(index=indices, columns='Key', values='Mean').reset_index()
    df_pivot['train_val_diff'] = df_pivot['train'] - df_pivot['val']
    df_pivot['val_test_diff'] = df_pivot['val'] - df_pivot['test']
    return df_pivot


def load_roc_data(root_dir):
    roc_data = {}

    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            # Extract the participant number and model name
            parts = dir_name.split('_')
            participant = parts[2]  # The <S> part
            model = parts[3]  # The <model> part

            if model not in roc_data:
                roc_data[model] = {}
            if participant not in roc_data[model]:
                roc_data[model][participant] = {}

            # Load the roc.pkl file
            roc_file = os.path.join(dir_path, 'roc.pkl')
            if os.path.isfile(roc_file):
                with open(roc_file, 'rb') as f:
                    roc_data[model][participant] = pickle.load(f)

    return roc_data
