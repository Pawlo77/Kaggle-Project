import sys
import os
import warnings
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from sklearn import set_config
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from tools import save_data, save_object, rename_cols, load_data

RANDOM_STATE = 42
warnings.filterwarnings("ignore")
set_config(transform_output="pandas")

# names of columns with nan values to be filled with "most_frequent"
norm_miss = ["job", "marital", "loan", "housing", "default"]
# will be filled according to random forest predictions
special_miss = ["education"]


def impute_normal(src="X_train.csv"):
    X_train = load_data(src)

    simple_imputer = ColumnTransformer(
        [
            ("fill", SimpleImputer(strategy="most_frequent"), norm_miss),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("simple_imputer", simple_imputer),
            ("rename1", FunctionTransformer(rename_cols)),
        ]
    )

    X_train = pipeline.fit_transform(X_train)

    save_data(X_train, "X_train_imputed_normal.csv")
    save_object(pipeline, "imputer_normal.pkl")


class SpecialImputer:
    def __init__(self, all, targets=special_miss):
        self.targets = targets
        self.learn_from = [col for col in all if col not in targets]
        self.models = []
        assert len(self.learn_from) > 0

    def fit(self, X, y=None):
        for col in self.targets:
            learn_bool = ~np.isnan(X.loc[:, col].to_numpy())
            learn_from = np.argwhere(learn_bool).ravel()
            X_learn = X.loc[learn_from, self.learn_from]
            y_learn = X.loc[learn_from, col]

            model = RandomForestClassifier(
                max_depth=6, n_jobs=-1, random_state=RANDOM_STATE
            )
            model.fit(X_learn, y_learn)
            self.models.append(model)

            score = accuracy_score(y_learn, model.predict(X_learn))
            print(f"Imputter accuracy score for seen data for {col}: {score:2.3f}")

        return self

    def transform(self, X, y=None):
        for i in range(len(self.targets)):
            fill_bool = np.isnan(X.loc[:, self.targets[i]].to_numpy())
            fill = np.argwhere(fill_bool).ravel()

            X_fill = X.loc[fill, self.learn_from]
            X.loc[fill, self.targets[i]] = self.models[i].predict(X_fill)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


def impute_special(src="X_train_preprocessed_cat.csv"):
    X_train = load_data(src)

    pipeline = SpecialImputer(X_train.columns)
    X_train = pipeline.fit_transform(X_train)

    save_data(X_train, "X_train_imputed_special.csv")
    save_object(pipeline, "imputer_special.pkl")
