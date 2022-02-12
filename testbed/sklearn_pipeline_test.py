import os
import sys
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

N_SAMPLES = 1000
N_FEATURES = 10

class MyRegressor(RandomForestRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict_neg(self, X):
        y_hat = super().predict(X)
        return np.concatenate([y_hat.reshape(-1, 1), -y_hat.reshape(-1, 1)], axis=1)

class ScoringOnlyRegressor(BaseEstimator):
    def __init__(self, trained_model):
        super().__init__()
        self.trained_model = trained_model

    # placeholder required by sklearn API spec
    def fit(self, X, y):
        pass

    def predict(self, X):
        y_hat = self.trained_model.predict_neg(X)
        return y_hat


def main():
    # create synthetic data for training and test
    X, y = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=8, random_state=123)

    df = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
    df.columns = [f'X{i:02d}' for i in range(N_FEATURES)] + ['y']


    train_df, test_df = train_test_split(df, test_size=0.2)
    print(train_df.shape, test_df.shape)

    # setup pipeline for training
    # setup column selector
    columns_to_use = ['X00', 'X01', 'X09']
    col_selector = ColumnTransformer(
        [('selector', 'passthrough', columns_to_use)]
    )
    col_selector.fit_transform(train_df)

    pipe1 = Pipeline(
        steps=[
            ('selector', col_selector),
            ('minmax', MinMaxScaler()),
        ]
    )

    pipe1.fit(train_df)
    y_hat1 = pipe1.transform(test_df)

    pipe2 = Pipeline(
        steps=[
            ('minmax', MinMaxScaler()),
        ]
    )

    columns_to_use2 = ['X00', 'X01', 'X02']
    pipe2.fit(train_df[columns_to_use])
    y_hat2 = pipe2.transform(test_df[columns_to_use])

    print(np.all(y_hat1 == y_hat2))

if __name__ == '__main__':
    main()