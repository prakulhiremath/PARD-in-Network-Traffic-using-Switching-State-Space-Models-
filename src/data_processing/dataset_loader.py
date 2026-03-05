import pandas as pd
import numpy as np


def load_dataset(path):

    df = pd.read_csv(path)

    # remove missing values
    df = df.dropna()

    # basic features
    features = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s"
    ]

    X = df[features].values

    # normalize
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X


def create_time_series(X, window=10):

    series = []

    for i in range(len(X) - window):
        series.append(X[i:i+window].mean(axis=0))

    return np.array(series)


if __name__ == "__main__":

    path = "data/raw/cicids2017.csv"

    X = load_dataset(path)

    ts = create_time_series(X)

    print("Dataset shape:", ts.shape)
