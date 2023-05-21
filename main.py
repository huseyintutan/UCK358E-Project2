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

# Filling the missing values

# Kategorik sütunları 'NA' ile doldurma
categorical_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']

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

# Concatenate encoded columns with the original data
data_encoded = pd.concat([data, encoded_cols], axis=1)

# Drop the original categorical columns
data_encoded.drop(categorical_cols, axis=1, inplace=True)

# Numeric columns for normalization
numeric_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'Age', 'TotalSF', 'HasGarage', 'TotalBath', 'TotalRooms', 'TotalVerandaArea']

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer([
    ('numeric_preprocessing', StandardScaler(), numeric_cols)
], remainder='passthrough')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_encoded, y, test_size=0.2, random_state=42)

# Create a pipeline with the preprocessor and linear regression model
pipeline = make_pipeline(preprocessor, LinearRegression())

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model using metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

# Predict target variable for the test dataset
test_encoded = pd.concat([test, encoded_cols], axis=1)
test_encoded.drop(categorical_cols, axis=1, inplace=True)

# Check for missing columns and add them with values set to 0
missing_cols = set(X_train.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0

test_pred = pipeline.predict(test_encoded)
predictions = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_pred})
predictions.to_csv('predictions.csv', index=False)
