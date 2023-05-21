#!/usr/bin/env python3

# @Author: Huseyin Tutan 110190021

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

data['Age'] = 2023 - data['GarageYrBlt']
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['HasGarage'] = data['GarageType'].apply(lambda x: 0 if x == 'NA' else 1)
data['TotalBath'] = data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])
data['TotalRooms'] = data['TotRmsAbvGrd'] + data['FullBath'] + data['HalfBath'] + data['BsmtFullBath'] + data['BsmtHalfBath']
data['TotalVerandaArea'] = data['WoodDeckSF'] + data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

X = data[features]
y = data['SalePrice']

# Eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
params = {
    'objective': ['reg:squarederror'],
    'learning_rate': [0.1],
    'max_depth': [3],
    'min_child_weight': [3],
    'gamma': [0],
    'subsample': [1.0],
    'colsample_bytree': [0.8],
    'eval_metric': ['rmse']
}
# params = {
#     'objective': ['reg:squarederror'],
#     'learning_rate': [0.1, 0.01],
#     'max_depth': [3, 5, 7],
#     'min_child_weight': [1, 3, 5],
#     'gamma': [0, 0.1, 0.2],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'eval_metric': ['rmse']
# }

model = xgb.XGBRegressor()

grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_

# print("En İyi Parametreler:")
# print(best_params)
# print("En İyi RMSE:", best_rmse)

model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE:", rmse)

test_X = test[features]
test_predictions = model.predict(test_X)

predictions = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_predictions})

predictions.to_csv('predictions.csv', index=False)
