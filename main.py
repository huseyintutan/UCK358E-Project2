#!/usr/bin/env python3

# @Author: Huseyin Tutan 110190021

# IMPORT THE LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# IMPORT THE DATA

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# MISSING VALUES

missing_values = data.isnull().sum()
missing_values_sorted = missing_values.sort_values(ascending=False)

missing_values_sorted_nonzero = missing_values_sorted[missing_values_sorted > 0]

plt.figure(figsize=(12, 6))
bar_plot = missing_values_sorted_nonzero.plot(kind='bar')
plt.title('Missing Values by Column')
plt.xlabel('Column')
plt.ylabel('Missing Value Count')

bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=90, ha='right')
plt.tight_layout()

# plt.show()

# Filling the missing values

# Kategorik sütunları 'NA' ile doldurma
categorical_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
non_numeric_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna('NA')
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].median())
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

# Feature Engineering

data['Age'] = 2023 - data['GarageYrBlt']
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['HasGarage'] = data['GarageType'].apply(lambda x: 0 if x == 'NA' else 1)
data['TotalBath'] = data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])
data['TotalRooms'] = data['TotRmsAbvGrd'] + data['FullBath'] + data['HalfBath'] + data['BsmtFullBath'] + data['BsmtHalfBath']
data['TotalVerandaArea'] = data['WoodDeckSF'] + data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']

# Selecting features and target variable
X = data[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'Age', 'TotalSF', 'HasGarage', 'TotalBath', 'TotalRooms', 'TotalVerandaArea']]
y = data['SalePrice']

# One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)

# Combine encoded categorical features with numerical features
X_encoded = pd.concat([X, encoded_cols], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=1459, random_state=42)

# Create and train the decision tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Feature importance
importance = model.feature_importances_
feature_names = X_encoded.columns
sorted_indices = np.argsort(importance)

plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_indices)), importance[sorted_indices], align='center')
plt.yticks(range(len(sorted_indices)), feature_names[sorted_indices])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Decision Tree Regressor - Feature Importance')

# plt.show()

# Make predictions on the test set
y_pred = model.predict(X_test)

# Create a DataFrame to store the predictions with house IDs
print.y_pred
predictions = pd.DataFrame({'Id': test['Id'], 'SalePrice': y_pred()})

# Save the predictions to a CSV file
predictions.to_csv('predictions.csv', index=False)
