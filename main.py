from scripts.imputting import impute
from scripts.preprocessing import preprocess_num, preprocess_cat
from scripts.tools import split_data


def main():
    print("Spliting data into training set and test set...")
    split_data()

    print("Imputting missing data...")
    impute()

    print("Preprocessing numerical data...")
    preprocess_num()

    print("Preprocessing categorical data...")
    preprocess_cat()


if __name__ == "__main__":
    main()
