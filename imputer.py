import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import set_config

from tools import get_data_path, save_data, save_object, rename_cols


RANDOM_STATE = 42
warnings.filterwarnings("ignore")
set_config(transform_output="pandas")


# only for categorical
class CustomTransformer(BaseEstimator, TransformerMixin):
    # train LabelEncoder on non nan entries
    def fit(self, X, y=None):
        self.label_encoders = {}
        self.columns = X.columns

        for col in X.columns:
            self.label_encoders[col] = LabelEncoder()
            x_c = X[col]

            self.label_encoders[col].fit(x_c[x_c.notnull()])

        return self

    # encode each value using trained LabelEncoders except entires with nan
    def transform(self, X, y=None):
        assert X.columns == self.columns

        for col in X.columns:
            x_c = X.loc[:, col]
            X.loc[:, col] = pd.Series(
                self.label_encoders[col].transform(x_c[x_c.notnull()]),
                index=x_c[x_c.notnull()].index,
            )

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


def load_data():
    data = pd.read_csv(
        get_data_path("bank-additional-full.csv"), sep=";", na_values="unknown"
    )

    X, y = data.drop("y", axis=1), data["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    save_data(X_train, "X_train.csv")
    save_data(X_test, "X_test.csv")
    save_data(y_train, "y_train.csv")
    save_data(y_test, "y_test.csv")

    return X_train, y_train


def main():
    X_train, _ = load_data()

    # names of categorical columns
    categorical = X_train.select_dtypes(include="object").columns
    # names of columns with nan values to be filled with "most_frequent"
    norm_miss = ["job", "marital", "loan", "housing", "default"]
    # names of columns with nan values to be filled based on random_forest predictions
    special_miss = ["education"]

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


if __name__ == "__main__":
    main()
