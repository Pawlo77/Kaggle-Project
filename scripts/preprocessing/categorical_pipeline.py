import pandas as pd
import numpy as np

from sklearn import set_config
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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


def make_pipeline(ordinal=[], one_hot=[]):
    ordinal_encoder = ColumnTransformer(
        generate_ordinal_encoders(ordinal),
        remainder="passthrough",
    )

    one_hot_encoder = ColumnTransformer(
        [("one_hot", OneHotEncoder(sparse_output=False), one_hot)],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("ordinal_encoder", ordinal_encoder),
            ("rename1", FunctionTransformer(rename_cols)),
            ("one_hot_encoder", one_hot_encoder),
            ("rename2", FunctionTransformer(rename_cols)),
        ]
    )

    return pipeline
