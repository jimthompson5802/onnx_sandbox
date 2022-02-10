import os
import sys
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X, y = make_regression(1000, 8, random_state=123)


df = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
df.columns = [f'X{i:02d}' for i in range(8)] + ['y']

train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)

column_set1 = ['X07', 'X02']
column_set2 = ['X07', 'X06', 'X01']


est1 = RandomForestRegressor(random_state=13)
est1.fit(train_df[column_set1], train_df['y'])
print(f'lr1 score: {est1.score(test_df[column_set1], test_df["y"])}')

est2 = RandomForestRegressor(random_state=31)
est2.fit(train_df[column_set2], train_df['y'])
print(f'lr2 score: {est2.score(test_df[column_set2], test_df["y"])}')


# setup pipeline for the pre-existing estimators


#  pipeline 1
selector1 = ColumnTransformer(
    [
        ('passthrough', 'passthrough', column_set1)
    ]
)
selector1.fit(train_df)


model1 = Pipeline([('selector', selector1), ('estimator', est1)])
print(f'model1 pipeline score: {model1.score(test_df, test_df["y"])}')


# pipeline 2
selector2 = ColumnTransformer(
    [
        ('passthrough', 'passthrough', column_set2)
    ]
)
selector2.fit(train_df)

model2 = Pipeline([('selector', selector2), ('estimator', est2)])
print(f'model2 pipeline score: {model2.score(test_df, test_df["y"])}')

est1_preds = est1.predict(test_df[column_set1])
est2_preds = est2.predict(test_df[column_set2])

model1_preds = model1.predict(test_df)
model2_preds = model2.predict(test_df)

assert np.all(est1_preds == model1_preds)
assert np.all(est2_preds == model2_preds)

print('all done')