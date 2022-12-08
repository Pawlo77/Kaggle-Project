from scripts.imputting import impute_normal, impute_special
from scripts.preprocessing import preprocess_num, preprocess_cat
from scripts.tools import split_data
from scripts.models import find_model


def main():
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

    print("Finding best model...")
    find_model()


if __name__ == "__main__":
    main()
