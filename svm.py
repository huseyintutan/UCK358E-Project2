#!/usr/bin/env python3

# @Author: Huseyin Tutan 110190021

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

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

# Eksik değerleri doldurma
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

model = SVR()

grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_rmse = np.sqrt(-grid_search.best_score_)

print("En İyi Parametreler:")
print(best_params)
print("En İyi RMSE:", best_rmse)

model = SVR(**best_params)
model.fit(X_train, y_train)

test_X = test[features]
test_X = imputer.transform(test_X)

test_predictions = model.predict(test_X)

predictions = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_predictions})

predictions.to_csv('predictions.csv', index=False)
