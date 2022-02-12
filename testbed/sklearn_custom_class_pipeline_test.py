import os
import sys
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

N_SAMPLES = 1000
N_FEATURES = 10

class MyRegressor(RandomForestRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict_neg(self, X):
        preds = super().predict(X)
        return np.concatenate([preds.reshape(-1, 1), -preds.reshape(-1, 1)], axis=1)


def main():
    # create synthetic data for training and test
    X, y = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=8, random_state=123)

    df = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
    df.columns = [f'X{i:02d}' for i in range(N_FEATURES)] + ['y']


    train_df, test_df = train_test_split(df, test_size=0.2)
    print(train_df.shape, test_df.shape)

    # train model
    rf = MyRegressor(random_state=123)
    rf.fit(df.drop(['y'], axis=1), df['y'])
    print(rf)

    print(rf.predict_neg(test_df.drop(['y'], axis=1))[:5])

    print(rf.predict(test_df.drop(['y'], axis=1)).reshape(-1, 1)[:5])


if __name__ == '__main__':
    main()