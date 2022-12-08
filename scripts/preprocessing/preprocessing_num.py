import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tools import get_data_path, save_data, save_object, load_data
from .preprocessing_tools import numerical_dist_plot
from .numerical_pipeline import NumericalPipeline


def preprocess_num(src="X_train_imputed_normal.csv"):
    X_train = load_data(src)
    y_train = load_data("y_train.csv")

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

    pipeline = NumericalPipeline(
        scale=scale,
        log_transform=log_transform,
        power_transform=power_transform,
        group=group,
        winsorize=winsorize,
    )
    X_train, y_train = pipeline.fit_transform(X_train, y=y_train, train=True)

    save_data(X_train, "X_train_preprocessed_num.csv")
    save_data(y_train, "y_train_f.csv")
    save_object(pipeline, "preprocessor_num.pkl")
    numerical_dist_plot(X_train)
