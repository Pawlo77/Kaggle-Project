import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from categorical_pipeline import make_pipeline
from tools import load_data, save_data, save_object


def preprocess_cat():
    X_train = load_data("X_train.csv")

    education_categories = [
        "illiterate",
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "professional.course",
        "university.degree",
    ]
    poutcome_categories = ["failure", "nonexistent", "success"]

    ordinal = [
        ("education", education_categories),
        ("poutcome", poutcome_categories),
    ]
    one_hot = [
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "contact",
    ]

    pipeline = make_pipeline(ordinal=ordinal, one_hot=one_hot)

    X_train = pipeline.fit_transform(X_train)

    save_data(X_train, "X_train_preprocessed_cat.csv")
    save_object(pipeline, "preprocessor_cat.pkl")


if __name__ == "__main__":
    preprocess_cat()
