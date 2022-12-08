import math
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import set_config

set_config(transform_output="pandas")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots")


def numerical_dist_plot(X_train):
    plt.figure(figsize=(20, 20))
    plt.tight_layout()

    numerical = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]

    for i, col in enumerate(numerical):
        plt.subplot(math.ceil(len(X_train.columns) // 4), 4, i + 1)
        plt.hist(X_train.loc[:, col], color="g", bins=100, alpha=1)
        plt.title(col)

    path = os.path.join(
        PLOT_DIR,
        "preprocessed_numerical_distributions.png",
    )
    plt.savefig(path)


def log_transform_col(x):
    return np.log1p(x)


def convert_names(names, remainder_num):
    if isinstance(names[0], tuple):
        return [("remainder__" * remainder_num + data[0], *data[1:]) for data in names]

    return ["remainder__" * remainder_num + name for name in names]
