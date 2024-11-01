import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from scipy.stats import sem
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import seaborn as sns
import numpy as np
import os

from data_analysis.data_utils import collect_results, process_df, compute_diff, load_roc_data


def bar_plot_dnl(df_new, d_val=None, n_val=None, save_path=None):

    # Ensure 'd', 'h', 'l', and 'Key' columns are in the DataFrame
    required_columns = {'d', 'n', 'l', 'Key'}
    if not required_columns.issubset(df_new.columns):
        raise ValueError(f"The dataframe must contain the following columns: {required_columns}")

    # Get unique values for d, h
    d_values = [d_val] if d_val is not None else df_new['d'].unique()
    n_values = [n_val] if n_val is not None else df_new['n'].unique()

    # Loop over all combinations of d and h
    for d in d_values:
        for n in n_values:
            # Filter the DataFrame for the current d and h values
            df_filtered = df_new[(df_new['d'] == d) & (df_new['n'] == n)]

            # Pivot the DataFrame to have 'l' as index and 'Key' as columns (val/test/train)
            df_pivot = df_filtered.pivot(index='l', columns='Key', values='Mean').reset_index()

            # Create a new figure for each combination of d and h
            fig, ax = plt.subplots(figsize=(12, 6))

            plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, zorder=0, color='0.9')

            # Bar plot width
            bar_width = 0.3

            # Calculate the x locations for the bars and the center positions
            x = np.arange(len(df_pivot['l']))
            x_middle = x + bar_width / 2  # Middle of the bars for proper x-ticks

            # Plot the bars for val and test
            bars_val = ax.bar(x, df_pivot['val'], width=bar_width, label='val', color='#f55f74', alpha=0.8, zorder=2)
            bars_test = ax.bar(x + bar_width, df_pivot['test'], width=bar_width, label='test', color='#298c8c',
                               alpha=0.8, zorder=2)

            # Plot lines connecting the middle of bars
            ax.plot(x_middle - bar_width / 2, df_pivot['val'], marker='o', color='#f55f74', label='_nolegend_',
                    linestyle='-', zorder=2)
            ax.plot(x_middle + bar_width / 2, df_pivot['test'], marker='o', color='#298c8c', label='_nolegend_',
                    linestyle='-', zorder=2)

            # Set plot title and labels
            ax.set_title(f'Val-Test Performance (D={d}, N={n})', fontsize=17)
            ax.set_xlabel('# of Layers', fontsize=17)
            ax.set_ylabel('ROC-AUC', fontsize=17)
            # ax.set_ylim([0.4, 0.7])

            # Set x-ticks to be in the middle of the bar groups
            ax.set_xticks(x_middle)
            ax.set_xticklabels(df_pivot['l'], rotation=0)

            # Add grid for both x and y axes
            # ax.grid(True)

            # Set legend for only the bars
            ax.legend(title='Type', fontsize=15, title_fontsize=17)

            ax.tick_params(axis='both', labelsize=17)

            plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
            # Optionally save the plot
            if save_path:
                plt.savefig(os.path.join(save_path, f'barplot_d{d}_n{n}.svg'))

            # Show the plot
            plt.show()


def line_plot_across_var(df_new, x, y, save_path=None):
    labels = {
        'l': '# of Layers',
        'o': 'Optimizer',
        'd': 'Feature Size',
        'n': '# of Training Samples',
        'h': '# of Hidden Units',
        'm': 'Model'
    }
    # df_pivot = compute_diff(df_new, indices=included_vars)
    df_pivot = df_new[df_new['Key'] == "val-test"]
    if len(included_vars) > 2:
        fixed_var = next((item for item in included_vars if item != x and item != y), None)
        unique_var_values = df_pivot[fixed_var].unique()
        colors = ['#800074', '#298c8c', '#f55f74', '#8cc5e3']
        for i, v in enumerate(unique_var_values):
            df_v = df_pivot[df_pivot[fixed_var] == v]

            # Sort the DataFrame by x and y to ensure correct plotting
            df_v = df_v.sort_values(by=[x, y])
            # Get the unique categories for y
            y_categories = df_v[y].unique()
            palette = {y_val: colors[i % len(colors)] for i, y_val in enumerate(y_categories)}
            df_v[y] = pd.Categorical(df_v[y], categories=sorted(y_categories))
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))

            sns.lineplot(data=df_v, x=x, y='Mean', hue=y, marker='o', palette=palette, ax=ax)
            # Shade the area around the mean using the SE
            for y_value in y_categories:
                df_y = df_v[df_v[y] == y_value]
                ax.fill_between(df_y[x],
                                df_y['Mean'] - df_y['SE'],
                                df_y['Mean'] + df_y['SE'],
                                color=palette[y_value],
                                alpha=0.3)  # Adjust alpha for shading transparency

            # Set the title and labels
            ax.set_title(f'Generalization Error Across {labels[x]} ({v})', fontsize=14)
            ax.set_xlabel(labels[x], fontsize=14)
            ax.set_ylabel('Generalization Error', fontsize=14)

            # Set x-axis to logarithmic scale with base 2
            ax.set_xscale('log', base=2)
            ax.grid(True)

            # Ensure the legend title and entries are correct
            handles, labels_ = ax.get_legend_handles_labels()
            labels_ = [l.capitalize() for l in labels_]
            ax.legend(handles=handles, labels=labels_, title=labels[y], fontsize=12)

            ax.tick_params(axis='both', labelsize=12)

            plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)

            # Optionally save the plot
            if save_path:
                plt.savefig(os.path.join(save_path, f'{fixed_var}_{v}.svg'))

            # Show the plot
            plt.show()


def line_plot_across_all_vars(df_new, x, y, save_path=None):
    df_pivot = compute_diff(df_new, indices=included_vars)

    if len(included_vars) > 2:
        fixed_var = next((item for item in included_vars if item != x and item != y), None)

        # Sort the DataFrame by x, y, and fixed_var to ensure correct plotting
        df_pivot = df_pivot.sort_values(by=[x, y, fixed_var])

        # Get the unique categories for y and fixed_var
        y_categories = df_pivot[y].unique()
        fixed_var_categories = df_pivot[fixed_var].unique()

        # Convert y and fixed_var to categorical variables with these categories
        df_pivot[y] = pd.Categorical(df_pivot[y], categories=sorted(y_categories))
        df_pivot[fixed_var] = pd.Categorical(df_pivot[fixed_var], categories=sorted(fixed_var_categories))

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Use 'hue' for different y values and 'style' for different fixed_var values (markers)
        sns.lineplot(data=df_pivot, x=x, y='val_test_diff', hue=y, style=fixed_var, markers=True, dashes=False, ax=ax)

        # Set the title and labels
        ax.set_title(f'Line Plot with {fixed_var} as markers')
        ax.set_xlabel(x)
        ax.set_ylabel('val_test_diff')

        # Set x-axis to logarithmic scale with base 2
        ax.set_xscale('log', base=2)
        ax.grid(True)

        # Ensure the legend title and entries are correct
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title=f'{y} / {fixed_var}', ncol=2)

        # Optionally save the plot
        if save_path:
            plt.savefig(os.path.join(save_path, f'combined_plot_{fixed_var}.png'))

        # Show the plot
        plt.show()


def line_plot_avg(df_new, x, y, save_path=None, **kwargs):
    labels = {
        'l': '# of Layers',
        'o': 'Optimizer',
        'd': 'Feature Size',
        # 'n': '# of Training Samples',
        'n': "Sample Size",
        'h': '# of Hidden Units',
        'f': 'Window Size',
        'ch': '# of Channels',
        'm': 'Architecture'
    }
    df_pivot = df_new[df_new['Key'] == "val-test"]
    df_pivot = df_pivot.pivot_table(index=included_vars, columns='Key', values='Mean').reset_index()
    if len(included_vars) > 2:

        # Compute average and standard deviation of 'val_test_diff' grouped by x and y
        grouped_df = df_pivot.groupby([x, y]).agg(
            val_test_diff_mean=('val-test', 'mean'),
            val_test_diff_std=('val-test', lambda a: a.std() / np.sqrt(len(a)))
        ).reset_index()

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, zorder=0, color='0.9')
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.02))

        colors = ['#f55f74', '#298c8c', '#800074', '#8cc5e3', "#f0c571", "#cecece"]
        if 'y_values' in kwargs.keys():
            y_values_all = kwargs['y_values']
        elif y == "l":
            y_values_all = [1, 4, 32, 64]
        elif y == "n":
            y_values_all = [50, 200, 400, 1600]
        elif y == "o":
            y_values_all = df_pivot[y].unique()
        else:
            raise NotImplementedError
        # Plot average values with standard deviation shaded area
        for i, y_value in enumerate(y_values_all):
            df_y = grouped_df[grouped_df[y] == y_value]
            ax.plot(df_y[x], df_y['val_test_diff_mean'], marker='o', color=colors[i], label=f'{y_value}')
            ax.fill_between(df_y[x],
                            df_y['val_test_diff_mean'] - df_y['val_test_diff_std'],
                            df_y['val_test_diff_mean'] + df_y['val_test_diff_std'],
                            alpha=0.3,
                            color=colors[i])  # Shaded area for standard deviation

        # Set the title and labels
        ax.set_title(f'Val-Test Gap Across {labels[x]}', fontsize=17)
        ax.set_xlabel(labels[x], fontsize=17)
        ax.set_ylabel('Val-Test Gap', fontsize=17)

        # Set x-axis to logarithmic scale with base 2
        ax.set_xscale('log', base=2)

        ax.tick_params(axis='both', labelsize=17)

        # Add legend
        ax.legend(title=labels[y], fontsize=16, title_fontsize=17, loc="best")

        plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.11)

        # Optionally save the plot
        if save_path:
            plt.savefig(os.path.join(save_path, f'combined_plot_with_avg_and_sd_{x}{y}.svg'))

        # Show the plot
        plt.show()


def plot_participant_roc(roc_data, save_path):
    for model, participants in roc_data.items():
        for participant, runs in participants.items():
            plt.figure(figsize=(10, 8))
            mean_tpr_val = 0.0
            mean_tpr_test = 0.0
            all_tpr_val = []
            all_tpr_test = []
            mean_fpr = np.linspace(0, 1, 10000)

            for run, data in runs.items():
                # Calculate ROC for validation set
                fpr_val, tpr_val, _ = roc_curve(data['y_true_val'], data['y_pred_val'])
                interp_tpr_val = np.interp(mean_fpr, fpr_val, tpr_val)
                mean_tpr_val += interp_tpr_val
                all_tpr_val.append(interp_tpr_val)

                # Calculate ROC for test set
                fpr_test, tpr_test, _ = roc_curve(data['y_true_test'], data['y_pred_test'])
                interp_tpr_test = np.interp(mean_fpr, fpr_test, tpr_test)
                mean_tpr_test += interp_tpr_test
                all_tpr_test.append(interp_tpr_test)

                # Plot individual ROC curves
                # plt.plot(fpr_val, tpr_val, color='blue', alpha=0.3)
                # plt.plot(fpr_test, tpr_test, color='red', alpha=0.3)

            # Calculate mean and std for validation and test
            mean_tpr_val /= len(runs)
            mean_tpr_test /= len(runs)
            std_tpr_val = np.std(all_tpr_val, axis=0)
            std_tpr_test = np.std(all_tpr_test, axis=0)

            # Plot mean ROC curves
            plt.plot(mean_fpr, mean_tpr_val, color='blue',
                     label=f'Mean Validation ROC (AUC = {auc(mean_fpr, mean_tpr_val):.2f})')
            plt.plot(mean_fpr, mean_tpr_test, color='red',
                     label=f'Mean Test ROC (AUC = {auc(mean_fpr, mean_tpr_test):.2f})')

            # Fill between for std
            plt.fill_between(mean_fpr, mean_tpr_val - std_tpr_val, mean_tpr_val + std_tpr_val, color='blue', alpha=0.2)
            plt.fill_between(mean_fpr, mean_tpr_test - std_tpr_test, mean_tpr_test + std_tpr_test, color='red',
                             alpha=0.2)

            plt.title(f'Participant {participant} - Model {model}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.grid(True)

            # Save the plot
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'{model}_participant_{participant}_roc.png'))
            plt.close()


def plot_aggregate_roc(roc_data, save_path):
    for model, participants in roc_data.items():
        plt.figure(figsize=(10, 8))
        mean_tpr_val = 0.0
        mean_tpr_test = 0.0
        all_tpr_val = []
        all_tpr_test = []
        mean_fpr = np.linspace(0, 1, 500)

        for participant, runs in participants.items():
            for run, data in runs.items():
                # Calculate ROC for validation set
                fpr_val, tpr_val, _ = roc_curve(data['y_true_val'], data['y_pred_val'])
                interp_tpr_val = np.interp(mean_fpr, fpr_val, tpr_val)
                mean_tpr_val += interp_tpr_val
                all_tpr_val.append(interp_tpr_val)

                # Calculate ROC for test set
                fpr_test, tpr_test, _ = roc_curve(data['y_true_test'], data['y_pred_test'])
                interp_tpr_test = np.interp(mean_fpr, fpr_test, tpr_test)
                mean_tpr_test += interp_tpr_test
                all_tpr_test.append(interp_tpr_test)

        # Calculate mean and SEM for validation and test
        mean_tpr_val /= len(participants) * len(runs)
        mean_tpr_test /= len(participants) * len(runs)
        sem_tpr_val = sem(all_tpr_val, axis=0)
        sem_tpr_test = sem(all_tpr_test, axis=0)

        # Plot mean ROC curves
        plt.plot(mean_fpr, mean_tpr_val, color='blue',
                 label=f'Mean Validation ROC (AUC = {auc(mean_fpr, mean_tpr_val):.2f})')
        plt.plot(mean_fpr, mean_tpr_test, color='red',
                 label=f'Mean Test ROC (AUC = {auc(mean_fpr, mean_tpr_test):.2f})')

        # Fill between for SEM
        plt.fill_between(mean_fpr, mean_tpr_val - sem_tpr_val, mean_tpr_val + sem_tpr_val, color='blue', alpha=0.2)
        plt.fill_between(mean_fpr, mean_tpr_test - sem_tpr_test, mean_tpr_test + sem_tpr_test, color='red', alpha=0.2)

        plt.title(f'Mean ROC Curve Across Participants - Model {model}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)

        # Save the plot
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{model}_aggregate_roc.png'))
        plt.close()


def box_plot_real_data(df_new, save_path=None):
    df_filtered = df_new[df_new['Key'].isin(['val', 'test'])]

    # Create a figure and axes
    plt.figure(figsize=(8, 6))

    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, zorder=0, color='0.93')

    # Prepare the data for plotting
    df_filtered['Key'] = df_filtered['Key'].astype('category')

    df_filtered = df_filtered.sort_values(by=['s'])

    # Plotting boxplots
    sns.boxplot(x='m', y='Mean', hue='Key', data=df_filtered, palette=['#298c8c', '#f55f74'], showfliers=False,
                saturation=0.9, zorder=2)

    # Customize xticks
    plt.tick_params(axis='both', labelsize=15)

    # Set labels and title
    plt.xlabel('Model', fontsize=15)
    plt.ylabel('ROC-AUC', fontsize=15)
    plt.title('Validation and Test Performances Across Subjects by Model', fontsize=15)

    # Add legend for 'val' and 'test'
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter out subject legends and keep only 'val' and 'test'
    handles = [handle for handle, label in zip(handles, labels) if label in ['val', 'test']]
    labels = [label for label in labels if label in ['val', 'test']]
    plt.legend(handles, labels, title='Type', fontsize=13, title_fontsize=14)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)

    # Show plot
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "val_test_real_eeg_all_subj.svg"))
    plt.show()


def bar_plot_across_var(df_new, x, y=None, save_path=None):
    unique_var_values = df_new[x].unique()

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
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                plot_mean_with_se(df_y, ax=ax)
                # plot_median_with_iqr(df_y, ax=ax[1])
                fig.suptitle(f"{x} = {v}, {y} = {y_val}", fontsize=18, y=0.97)
                plt.subplots_adjust(top=0.9)
                if save_path:
                    plt.savefig(os.path.join(save_path, f"{x}{v}_{y}{y_val}.png"))
                plt.close()
        else:
            # Sort by the second variable if it exists
            if second_var is not None:
                df_v = df_v.sort_values(by=second_var)
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            plot_mean_with_se(df_v, ax=ax)
            # plot_median_with_iqr(df_v, ax=ax[1])
            if x == 'd':
                fig.suptitle(f"data dimensionality = {v}", fontsize=18, y=0.97)
            elif x == 'r':
                fig.suptitle(f"ratio of large eigen values = {v}", fontsize=18, y=0.97)
            elif x == 'h':
                fig.suptitle(f"hidden size = {v}", fontsize=18, y=0.97)
            elif x == 'l':
                fig.suptitle(f"number of layers = {v}", fontsize=18, y=0.97)
            elif x == 'n':
                fig.suptitle(f"number of training samples = {v}", fontsize=18, y=0.97)
            plt.subplots_adjust(top=0.9)
            if save_path:
                plt.savefig(os.path.join(save_path, f"{x}{v}.png"))
            plt.close()


def plot_mean_with_se(df, ax=None, save_path=None, figname="test.png"):  # bar plot
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=str, default='d')
    parser.add_argument('-y', type=str, default=None)
    parser.add_argument('--y_vals', nargs='+', type=int, default=None)
    parser.add_argument('--fixed_vars', nargs='+', type=str, default=None)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--task', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment = args.experiment
    task = args.task
    root_path = args.root_path
    data_dir = os.path.join(root_path, experiment)
    save_dir = os.path.join(root_path, experiment + "_plots")
    os.makedirs(save_dir, exist_ok=True)
    results = collect_results(data_dir)
    if task == "bar_plot_fixed_var":
        dff, included_vars = process_df(results)
        bar_plot_across_var(dff, x=args.x, y=args.y, save_path=save_dir)
    elif task == "line_plot_averaged":
        dff, included_vars = process_df(results)
        if args.y_vals is not None:
            line_plot_avg(dff, x=args.x, y=args.y, save_path=save_dir, y_values=args.y_vals)
        else:
            line_plot_avg(dff, x=args.x, y=args.y, save_path=save_dir)
    elif task == "line_plot_fixed_var":
        df, included_vars = process_df(results)
        line_plot_across_var(df, x=args.x, y=args.y, save_path=save_dir)
    elif task == "bar_plot_dnl":
        dff, included_vars = process_df(results)
        bar_plot_dnl(dff, d_val=64, n_val=50, save_path=save_dir)
    elif task == "synth_eeg":
        # line_plot_across_var(results, x='n', y='m', save_path=save_dir)
        dff, included_vars = process_df(results)
        fixed_vars = args.fixed_vars
        # dff = dff[dff[fixed_vars[0]] == 'resnet1d']
        # dff = dff[dff[fixed_vars[1]] == 64]
        line_plot_avg(dff, x=args.x, y=args.y, save_path=save_dir)
    elif task == "real_eeg":
        roc_content = load_roc_data(root_dir=data_dir)
        dff, included_vars = process_df(results)
        # plot_participant_roc(roc_content, save_dir)
        # plot_aggregate_roc(roc_content, save_dir)
        box_plot_real_data(dff, save_path=save_dir)
    else:
        raise NotImplementedError
