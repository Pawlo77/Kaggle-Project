import sys
import os
import pandas as pd
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from sklearn import set_config
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from tools import save_data, save_object, rename_cols, load_data
from custom_transformer import CustomTransformer

RANDOM_STATE = 42
warnings.filterwarnings("ignore")
set_config(transform_output="pandas")

# names of columns with nan values to be filled with "most_frequent"
norm_miss = ["job", "marital", "loan", "housing", "default"]
# names of columns with nan values to be filled based on random_forest predictions
special_miss = ["education"]


def impute_normal():
    X_train = load_data("X_train.csv")

    # names of categorical columns
    categorical = X_train.select_dtypes(include="object").columns

    pp = Pipeline(
        [
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("transform", OrdinalEncoder()),
        ]
    )
    ct = ColumnTransformer(
        [
            ("norm", pp, norm_miss),
            (
                "special",
                CustomTransformer(),
                special_miss,
            ),
            # encode remaining categorical features
            (
                "rest",
                OrdinalEncoder(),
                [col for col in categorical if col not in special_miss + norm_miss],
            ),
        ],
    )

    special_imputer = IterativeImputer(
        estimator=RandomForestClassifier(max_depth=6, n_jobs=-1),
        initial_strategy="most_frequent",
        random_state=RANDOM_STATE,
        max_iter=30,
    )

    pipeline = Pipeline(
        [
            # prepare categorical features
            ("ct", ct),
            # use categorical dataset to predict missing values in education and default
            ("special_imputer", special_imputer),
            ("rename", FunctionTransformer(rename_cols)),
        ]
    )

    cat = X_train[categorical]
    num = X_train.drop(categorical, axis=1)

    cat = pipeline.fit_transform(cat)

    # # merge cat and num array together
    X_train = pd.concat([num, cat], axis=1, copy=False)

    save_data(X_train, "X_train_imputed.csv")
    save_object(ct, "imputer.pkl")


def impute_special():
    pass


if __name__ == "__main__":
    impute_normal()
    impute_special()
