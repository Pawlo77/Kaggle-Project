from sklearn import set_config
from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
    PowerTransformer,
    KBinsDiscretizer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from tools import rename_cols
from .preprocessing_tools import log_transform_col
from .winsorizer import Winsorizer

set_config(transform_output="pandas")


class NumericalPipeline:
    def __init__(
        self, scale=[], log_transform=[], power_transform=[], group=[], winsorize=[]
    ):
        self.scaler = ColumnTransformer(
            [("scaler", StandardScaler(), scale)], remainder="passthrough"
        )

        self.log_transformer = ColumnTransformer(
            [
                (
                    "log_transformer",
                    FunctionTransformer(log_transform_col),
                    log_transform,
                )
            ],
            remainder="passthrough",
        )

        self.power_transformer = ColumnTransformer(
            [("power_transformer", PowerTransformer(), power_transform)],
            remainder="passthrough",
        )

        self.grouper = ColumnTransformer(
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

        self.winsorizer = Winsorizer(columns=winsorize)
        self.pipeline = Pipeline(
            [
                ("scaler", self.scaler),
                ("rename1", FunctionTransformer(rename_cols)),
                ("log_transformer", self.log_transformer),
                ("rename2", FunctionTransformer(rename_cols)),
                ("power_transformer", self.power_transformer),
                ("rename3", FunctionTransformer(rename_cols)),
                ("grouper", self.grouper),
                ("rename5", FunctionTransformer(rename_cols)),
            ]
        )

    def fit(self, X, y=None):
        self.winsorizer.fit(X, y)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X, y=None):
        X = self.winsorizer.transform(X)
        X = self.pipeline.transform(X)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
