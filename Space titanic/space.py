# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def cabinUpdate(X):
    deck = [];
    port = [];
    for i in X.Cabin:
        # print(i)
        if isinstance(i, str):
            deck.append(i[0])
            port.append(i[-1])
        else:
            deck.append('A')
            port.append('S')
    X=X.drop(['Cabin'], axis = 1)
    X=X.join(pd.DataFrame({'Deck': deck,'Port': port}, index=X.index))
    return X

# Read the data
X = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv', index_col='PassengerId')
X_test_full = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv', index_col='PassengerId')
X=X.drop(['Name'], axis=1)
X = cabinUpdate(X)
X_test_full=X_test_full.drop(['Name'], axis=1)
X_test_full = cabinUpdate(X_test_full)


# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['Transported'], inplace=True)
y = X.Transported              
X.drop(['Transported'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)


my_model = XGBRegressor(n_estimators=1500, learning_rate=0.05, random_state=0) # Your code here
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

pred = my_model.predict(X_valid)
for i in range(len(pred)):
    pred[i]=round(pred[i])

mae = mean_absolute_error(pred, y_valid)
print(mae)

# print(y)
# for i in range(10):
#     print(pred[i])

# print(X_test.head())
preds = my_model.predict(X_test)
for i in range(len(preds)):
    preds[i]=round(preds[i])

data_to_submit = pd.DataFrame({
    'PassengerId': X_test.index,
    'Transported': preds.astype(bool)
})
data_to_submit.to_csv('submission.csv', index=False)