import pandas as pd
import numpy as np


# only for numerical, accepts only a pd.DataFrame
class Winsorizer:
    def __init__(self, columns, lower=0.005, upper=0.005):
        self.columns = columns
        self.lower = lower
        self.upper = upper
        self.percentiles = {}

    # train LabelEncoder on non nan entries
    def fit(self, X, y=None):
        for col in self.columns:
            self.percentiles[col] = np.percentile(
                X.loc[:, col],
                [
                    self.lower * 100,
                    (1 - self.upper) * 100,
                ],
            )

        return self

    # encode each value using trained LabelEncoders except entires with nan
    def transform(self, X, y=None):
        to_remove = set()

        for col in self.columns:
            lower, upper = self.percentiles[col]

            for i in range(X.shape[0]):
                if not lower < X.loc[i, col] < upper:
                    to_remove.add(i)

        return X.drop(to_remove, axis=0)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
