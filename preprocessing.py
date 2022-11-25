import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
    PowerTransformer,
    KBinsDiscretizer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from scipy.stats.mstats import winsorize

from tools import get_data_path, rename_cols, save_data, save_object

set_config(transform_output="pandas")


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
    plt.savefig("preprocessed_numerical_distributions.png")


def winsorize_col(col):
    return pd.DataFrame(
        winsorize(col.to_numpy(), limits=[0.05, 0.05], axis=1),
        index=col.index,
        columns=col.columns,
    )


def log_transform_col(x):
    return np.log(x + 1)


def convert_names(names, remainder_num):
    if isinstance(names[0], tuple):
        return [("remainder__" * remainder_num + data[0], *data[1:]) for data in names]

    return ["remainder__" * remainder_num + name for name in names]


def main():
    X_train = pd.read_csv(get_data_path("X_train_imputed.csv"))

    scale = ["duration", "campaign", "nr.employed"]
    log_transform = ["duration"]
    power_transform = ["campaign", "age"]
    winsorize = ["age", "cons.price.idx", "cons.conf.idx", "euribor3m"]
    group = [
        # ("age", 10, "uniform"),
        ("euribor3m", 4, "kmeans"),
        ("cons.conf.idx", 5, "kmeans"),
        ("cons.price.idx", 6, "kmeans"),
        ("previous", 2, "kmeans"),
        ("pdays", 2, "kmeans"),
        ("emp.var.rate", 4, "kmeans"),
    ]

    scaler = ColumnTransformer(
        [("scaler", StandardScaler(), scale)], remainder="passthrough"
    )
    log_transformer = ColumnTransformer(
        [("log_transformer", FunctionTransformer(log_transform_col), log_transform)],
        remainder="passthrough",
    )
    power_transformer = ColumnTransformer(
        [("power_transformer", PowerTransformer(), power_transform)],
        remainder="passthrough",
    )
    winsorizer = ColumnTransformer(
        [("winsorizer", FunctionTransformer(winsorize_col), winsorize)],
        remainder="passthrough",
    )
    grouper = ColumnTransformer(
        [
            (
                f"grouper_{col}",
                KBinsDiscretizer(encode="ordinal", strategy=strategy, n_bins=bins),
                [col],
            )
            for col, bins, strategy in group
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("rename1", FunctionTransformer(rename_cols)),
            ("log_transformer", log_transformer),
            ("rename2", FunctionTransformer(rename_cols)),
            ("power_transformer", power_transformer),
            ("rename3", FunctionTransformer(rename_cols)),
            ("winsorize", winsorizer),
            ("rename4", FunctionTransformer(rename_cols)),
            ("grouper", grouper),
            ("rename5", FunctionTransformer(rename_cols)),
        ]
    )

    X_train = pipeline.fit_transform(X_train)

    save_data(X_train, "X_train_preprocessed.csv")
    save_object(pipeline, "preprocessor.pkl")
    numerical_dist_plot(X_train)


if __name__ == "__main__":
    main()
