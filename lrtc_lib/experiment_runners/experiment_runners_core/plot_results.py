# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_palette("tab10")


def plot_metric(dataset, metric, df, output_path):
    """
    Plot the results in the most basic fashion. No complex color markers labels etc.
    """
    models = sorted(df["model"].unique())
    als = sorted(df["AL"].unique())
    x_col = "train total count"
    for model in models:
        for al in als:
            if al == "no_active_learning":
                continue
            model_df_all = df[(df["model"] == model) & ((df["AL"] == al) | (df["AL"] == "no_active_learning"))]
            model_df = model_df_all[[x_col, metric]]
            model_df = model_df.dropna(axis=0)
            model_df = model_df.groupby(x_col, as_index=False).mean()
            x = model_df[x_col]
            y = model_df[metric]
            ax = sns.lineplot(x=x, y=y, label=f'{al}_{model}')
    plt.title(dataset)
    plt.savefig(os.path.join(output_path, f"{dataset}_{metric}"))
    plt.show()
    plt.close()


def plot_results(path):
    df = pd.read_csv(path)
    datasets = df["dataset"].unique()
    metrics = ["accuracy"]
    for dataset in datasets:
        sub_df = df[df["dataset"] == dataset]
        for metric in metrics:
            plot_metric(dataset, metric, sub_df, Path(path).parent)


if __name__ == '__main__':
    from lrtc_lib.definitions import ROOT_DIR
    results_file_path = "output/experiments/balanced_NB_20201020_1730/results/balanced_NB_all_repeats.csv"
    plot_results(os.path.join(ROOT_DIR, results_file_path))
