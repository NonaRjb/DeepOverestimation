import math
import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from data_analysis.data_utils import collect_results, process_df


if __name__ == "__main__":
    experiment = "N1000_dhl"
    root_path = "/Volumes/T5 EVO/Overfitting/out/loss/"
    data_dir = os.path.join(root_path, experiment)
    save_dir = os.path.join(root_path, experiment+"_plots")
    results = collect_results(data_dir)
    df, included_vars = process_df(results)
    # Pivot the dataframe to have separate columns for train, val, and test
    df_pivot = df.pivot_table(index=['d', 'l', 'h'], columns='Key', values='Mean').reset_index()

    # Calculate performance differences
    df_pivot['train_val_diff'] = df_pivot['train'] - df_pivot['val']
    df_pivot['val_test_diff'] = df_pivot['val'] - df_pivot['test']

    # Choose which performance difference to analyze
    y = df_pivot['val_test_diff']  # or df_pivot['train_val_diff']

    # Independent variables
    X = df_pivot[['d', 'l', 'h']]
    X = sm.add_constant(X)  # Add constant for intercept

    # Perform linear regression
    model = sm.OLS(y, X).fit()

    # Print the results
    print(model.summary())

    # Scatter plot of one parameter vs performance difference
    # fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    # axs[0].scatter(df_pivot['d'], df_pivot['val_test_diff'])
    # axs[0].set_xlabel('Feature Dimensionality (d)')
    # axs[0].set_ylabel('Validation - Test Performance Difference')
    #
    # axs[1].scatter(df_pivot['h'], df_pivot['val_test_diff'])
    # axs[1].set_xlabel('Hidden Size (H)')
    # axs[1].set_ylabel('Validation - Test Performance Difference')
    #
    # axs[2].scatter(df_pivot['r'], df_pivot['val_test_diff'])
    # axs[2].set_xlabel('Ratio of Large EVs (r)')
    # axs[2].set_ylabel('Validation - Test Performance Difference')
    #
    # plt.show()

    var_to_plot = 'l'
    constant_d_value = 256
    # constant_r_value = 2**(-5)
    constant_h_value = 8
    constant_l_value = 2
    df_subset = df_pivot[(df_pivot['d'] == constant_d_value) & (df_pivot['h'] == constant_h_value)]

    # Regression on subset 2 (h vs performance_diff)
    X2 = np.log2(df_subset[['l']])
    y2 = df_subset['val_test_diff']
    X2 = sm.add_constant(X2)  # Adds the intercept term to the model
    model2 = sm.OLS(y2, X2).fit()

    print(model2.summary())

    pred_ols = model2.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]

    fig2, ax = plt.subplots(figsize=(8, 6))

    ax.plot(X2[var_to_plot], y2, "o", label="data")
    ax.plot(X2[var_to_plot], model2.fittedvalues, "r--.", label="OLS")
    ax.plot(X2[var_to_plot], iv_u, "r--")
    ax.plot(X2[var_to_plot], iv_l, "r--")
    ax.legend(loc="best")

    plt.show()
