import argparse
import os
import warnings

import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    input_data_path = os.path.join("/opt/ml/processing/input", "Bank.csv")

    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)

    split_ratio = args.train_test_split_ratio
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("y", axis=1), df["y"], test_size=split_ratio, random_state=0
    )

    train_output_path = os.path.join("/opt/ml/processing/output/train", "train.csv")
    print("Saving training to {}".format(train_output_path))
    pd.DataFrame(X_train).to_csv(train_output_path, header=False, index=False)

    test_output_path = os.path.join("/opt/ml/processing/output/test", "test.csv")
    print("Saving test to {}".format(test_output_path))
    pd.DataFrame(X_test).to_csv(test_output_path, header=False, index=False)
