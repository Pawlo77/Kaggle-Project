import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from .categorical_pipeline import CategoricalPipeline
from tools import load_data, save_data, save_object


def preprocess_cat(src="X_train_preprocessed_num.csv"):
    X_train = load_data(src)
    y_train = load_data("y_train_f.csv")

    education_categories = [
        "illiterate",
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "professional.course",
        "university.degree",
    ]
    month_categories = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    day_of_week_categories = ["mon", "tue", "wed", "thu", "fri"]

    poutcome_categories = ["failure", "nonexistent", "success"]

    ordinal = [
        ("education", education_categories),
        ("poutcome", poutcome_categories),
        ("month", month_categories),
        ("day_of_week", day_of_week_categories),
        # ("default", "auto"),
    ]
    one_hot = [
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "contact",
    ]
    sin_cos = ["month", "day_of_week"]

    pipeline = CategoricalPipeline(ordinal=ordinal, one_hot=one_hot, sin_cos=sin_cos)

    X_train, y_train = pipeline.fit_transform(X_train, y_train)

    save_data(X_train, "X_train_preprocessed_cat.csv")
    save_data(y_train, "y_train_f.csv")
    save_object(pipeline, "preprocessor_cat.pkl")
