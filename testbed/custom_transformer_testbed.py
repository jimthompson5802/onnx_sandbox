import os
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


N_SAMPLES = 10
N_FEATURES = 3


class DFSimpleImputer(SimpleImputer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X):
        if self.indicator_:
            return pd.DataFrame(
                super().transform(X),
                columns=self.feature_names_in_.tolist() + [f'j{c}' for c in self.feature_names_in_]
            )
        else:
            return pd.DataFrame(super().transform(X), columns=self.feature_names_in_)


class DFNearbyImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy:str='median', fill_value:float=None, add_indicator:bool=False,
                 groupby:str=None) -> None:
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.groupby = groupby
        self.feature_names_in_ = None
        self.nearby_imputed_values_ = None
        self.overall_imputed_values_ = None

        self.summary_stats = {
            'median': pd.DataFrame.median,
            'mean': pd.DataFrame.mean,
            'constant': self._fill_in_constant
        }

    def _fill_in_constant(self, X):
        # drop the groupby column
        column_names = X.drop([self.groupby], axis=1).columns
        values = np.ones([len(column_names)]) * self.fill_value
        return pd.Series({k:v for k, v in zip(column_names, values)})

    def fit(self, X:pd.DataFrame, y=None) -> object:
        self.feature_names_in_ = X.drop([self.groupby], axis=1).columns.to_list()
        self.nearby_imputed_values_ = X.groupby([self.groupby]).apply(self.summary_stats[self.strategy],
                                                               numeric_only=True)
        self.overall_imputed_values_ = X.drop([self.groupby], axis=1).apply(self.summary_stats[self.strategy])
        print(f"nearby:\n{self.nearby_imputed_values_}")
        print(f"overall:\n{self.overall_imputed_values_}")
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in df.drop([self.groupby], axis=1).columns:
            for group in X[self.groupby].unique():
                mask = (X[col].notna()) | (X[self.groupby] != group)
                try:
                    # get summary statistic for "nearby" observations
                    df.loc[:, col] = df.loc[:, col].where(mask, other=self.nearby_imputed_values_.loc[group, col])
                except KeyError:
                    # if no summary statistic exists for the "neraby" location, use overall summary stats
                    df.loc[:, col] = df.loc[:, col].where(mask, other=self.overall_imputed_values_.loc[col])
        if self.add_indicator:
            missing_indicator = X.drop([self.groupby], axis=1).isna()
            df = pd.DataFrame(
                np.concatenate([df, np.int8(missing_indicator)], axis=1),
                columns=[
                    df.columns.to_list() + [f"j{c}" for c in df.drop([self.groupby], axis=1).columns]
                ]
            )

        return df

def main():
    X, _ = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=123)

    # create some missing values
    np.random.seed(123)
    idx = np.random.choice(range(N_SAMPLES), size=int(0.5 * N_SAMPLES))
    X[idx, 0] = np.nan
    idx = np.random.choice(range(N_SAMPLES), size=int(0.3 * N_SAMPLES))
    X[idx, 1] = np.nan
    idx = np.random.choice(range(N_SAMPLES), size=int(0.5 * N_SAMPLES))
    X[idx, 2] = np.nan

    ctract = np.random.choice(['a', 'b'], size=N_SAMPLES, replace=True).reshape(-1, 1)

    df = pd.DataFrame(np.concatenate([X, ctract], axis=1), columns=['X00', 'X01', 'X02', 'ctract'])
    df = df.astype({'X00': np.float32, 'X01': np.float32, 'X02': np.float32})

    print(df.dtypes)
    print(df)

    pipe = make_pipeline(DFNearbyImputer(strategy='median', groupby='ctract', add_indicator=True))
    print(pipe.fit_transform(df))
    print(pipe.transform(df).dtypes)

    # add unknown ctract to data
    df_test = pd.concat(
        [
            df,
            pd.DataFrame([{'X00': np.nan, 'X01': np.nan, 'X02': np.nan, 'ctract':'c'}])
        ],
        ignore_index=True
    )
    print(df_test)
    print(df_test.dtypes)
    print(pipe.transform(df_test))
    print(pipe.transform(df_test).dtypes)

    print('all done')

if __name__ == '__main__':
    main()