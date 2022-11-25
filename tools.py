import os
import pickle
import pandas as pd


def get_data_path(file):
    return os.path.join("data", file)


def save_data(df, name, header=True):
    path = get_data_path(name)

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df.to_csv(path, index=False, header=header)


def save_object(obj, name):
    path = os.path.join("pickles", name)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def rename_col(col):
    return col.split("__")[-1]


def rename_cols(cols):
    return cols.rename(rename_col, axis="columns")
