import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PICKLES_PATH = os.path.join(os.path.dirname(__file__), "..", "pickles")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
RANDOM_STATE = 42

os.makedirs(PICKLES_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)


def get_data_path(name):
    return os.path.join(DATA_PATH, name)


def save_data(df, name, header=True):
    path = get_data_path(name)

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df.to_csv(path, index=False, header=header)


def load_data(name, **kwargs):
    path = get_data_path(name)
    return pd.read_csv(path, **kwargs)


def save_object(obj, name):
    path = os.path.join(PICKLES_PATH, name)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(name):
    path = os.path.join(PICKLES_PATH, name)
    with open(path, "rb") as f:
        return pickle.load(f)


def rename_col(col):
    return col.split("__")[-1]


def rename_cols(cols):
    return cols.rename(rename_col, axis="columns")


def split_data(random_state=RANDOM_STATE):
    data = pd.read_csv(
        get_data_path("bank-additional-full.csv"), sep=";", na_values="unknown"
    )

    X, y = data.drop("y", axis=1), data["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    save_data(X_train, "X_train.csv")
    save_data(X_test, "X_test.csv")
    save_data(y_train, "y_train.csv")
    save_data(y_test, "y_test.csv")


class MyPipeline(Pipeline):
    def transform(self, X, y=None):
        X = super().transform(X)
        return X, y


class DataPipeline:
    def __init__(
        self,
        steps=[
            "imputer_normal",
            "preprocessor_num",
            "preprocessor_cat",
            "imputer_special",
        ],
    ):
        self.pipelines = [load_object(name + ".pkl") for name in steps]

    def transform(self, X, y=None):
        for pipeline in self.pipelines:
            X, y = pipeline.transform(X, y)

        return X, y
