import numpy as np

from sklearn import set_config
from sklearn.preprocessing import (
    OrdinalEncoder,
    FunctionTransformer,
    OneHotEncoder,
    MinMaxScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from tools import rename_cols

set_config(transform_output="pandas")


def generate_ordinal_encoders(ordinal):
    ordinal_names = [o[0] for o in ordinal]
    # at each position list of unique categories or "auto"
    ordinal_categories = [o[1] for o in ordinal]

    return [
        (
            f"ordinal_{i}",
            OrdinalEncoder(
                categories=[ordinal_categories[i]]
                if ordinal_categories[i] != "auto"
                else "auto",
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            ),
            [ordinal_names[i]],
        )
        for i in range(len(ordinal_names))
    ]


class CategoricalPipeline:
    def __init__(self, ordinal=[], one_hot=[], sin_cos=[]):
        self.ordinal_encoder = ColumnTransformer(
            generate_ordinal_encoders(ordinal),
            remainder="passthrough",
        )

        self.one_hot_encoder = ColumnTransformer(
            [("one_hot", OneHotEncoder(sparse_output=False), one_hot)],
            remainder="passthrough",
        )

        self.sin_cos_encoder_helper = ColumnTransformer(
            [("scaler", MinMaxScaler(feature_range=(0, 2 * np.pi)), sin_cos)],
            remainder="passthrough",
        )
        self.sin_cos = sin_cos

        self.pipeline = Pipeline(
            [
                ("ordinal_encoder", self.ordinal_encoder),
                ("rename1", FunctionTransformer(rename_cols)),
                ("one_hot_encoder", self.one_hot_encoder),
                ("rename2", FunctionTransformer(rename_cols)),
                ("sin_cos_helper", self.sin_cos_encoder_helper),
                ("rename3", FunctionTransformer(rename_cols)),
            ]
        )

        self.y_pipeline = Pipeline(
            [
                ("ordinal_encoder", OrdinalEncoder()),
                ("rename1", FunctionTransformer(rename_cols)),
            ]
        )

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        self.y_pipeline.fit(y)
        return self

    def transform(self, X, y=None):
        X = self.pipeline.transform(X)
        y = self.y_pipeline.transform(y)

        for col in self.sin_cos:
            X[f"{col}_sin"] = np.sin(X.loc[:, col])
            X[f"{col}_cos"] = np.cos(X.loc[:, col])
        X.drop(self.sin_cos, axis=1, inplace=True)

        return X, y

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
