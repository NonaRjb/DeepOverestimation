import matplotlib.pyplot as plt
import numpy as np
import os

from data_analysis.data_utils import collect_results, process_df


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
    experiment = "n5000_dhr_3_rep"
    root_path = "/Volumes/T5 EVO/Overfitting/out/loss/"
    data_dir = os.path.join(root_path, experiment)
    save_dir = os.path.join(root_path, experiment+"_plots")
    results = collect_results(data_dir)
    plot_across_var(results, x='d', y='r', save_path=save_dir)
