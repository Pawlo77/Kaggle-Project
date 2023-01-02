import os
import warnings

from scripts.imputting import impute_normal, impute_special
from scripts.preprocessing import preprocess_num, preprocess_cat
from scripts.tools import (
    split_data,
    DataPipeline,
    save_object,
    load_object,
    load_data,
    save_data,
    get_data_path,
)
from scripts.models import find_model


def main(ignore_warnings=True):
    if ignore_warnings:
        warnings.simplefilter("ignore")

    print("Spliting data into training set and test set...")
    split_data()

    print("Naive imputting missing data...")
    impute_normal()

    print("Preprocessing numerical data...")
    preprocess_num()

    print("Preprocessing categorical data...")
    preprocess_cat()

    print("Smart imputting missing data...")
    impute_special()

    print("Generating data pipeline...")
    pipeline = DataPipeline()
    save_object(pipeline, "data_pipeline.pkl")

    print("Preprocessing test data and renaming X_train...")
    pipeline = load_object("data_pipeline.pkl")
    X_test = load_data("X_test.csv")
    y_test = load_data("y_test.csv")
    X_test, y_test = pipeline.transform(X_test, y_test)
    save_data(X_test, "X_test_f.csv")
    save_data(y_test, "y_test_f.csv")
    os.rename(
        get_data_path("X_train_imputed_special.csv"), get_data_path("X_train_f.csv")
    )

    print("Finding best model...")
    find_model()


if __name__ == "__main__":
    main()
