import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


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
        assert (X.columns == self.columns).all()

        for col in X.columns:
            x_c = X.loc[:, col]
            X.loc[:, col] = pd.Series(
                self.label_encoders[col].transform(x_c[x_c.notnull()]),
                index=x_c[x_c.notnull()].index,
            )

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns
