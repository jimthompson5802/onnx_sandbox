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
    columns_to_use = ['X00', 'X09', 'X03']
    col_selector = ColumnTransformer(
        [('selector', 'passthrough', columns_to_use)]
    )
    col_selector.fit_transform(train_df)

    # train model
    rf = MyRegressor(random_state=123)
    t_pipe = Pipeline(steps=[
        ('selector', col_selector),
        ('estimator', rf)
    ])

    t_pipe.fit(df.drop(['y'], axis=1), df['y'])
    print(t_pipe)

    print(rf.predict_neg(test_df[columns_to_use])[:5])

    print(rf.predict(test_df[columns_to_use]).reshape(-1,1)[:5])

    # setup proxy pipeline for scoring
    rf2 = ScoringOnlyRegressor(rf)
    s_pipe = Pipeline(
        steps = [
            ('selector', col_selector),
            ('estimator', rf2)
        ]
    )
    print(s_pipe.predict(test_df)[:5])


if __name__ == '__main__':
    main()