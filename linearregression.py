import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data by imputing missing values with mean
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Define the parameter grid
params = {'fit_intercept': [True, False]}

# Create the linear regression model
model = LinearRegression()

# Perform grid search to find the best parameters
grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters and the best RMSE score
best_params = grid_search.best_params_
best_rmse = np.sqrt(-grid_search.best_score_)

print("Best Parameters:")
print(best_params)
print("Best RMSE:", best_rmse)

# Create a new model with the best parameters
model = LinearRegression(**best_params)
model.fit(X_train, y_train)

# Preprocess the test data
test_X = imputer.transform(test[features])

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE:", rmse)

# Make predictions on the test dataset
test_predictions = model.predict(test_X)

predictions = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_predictions})

predictions.to_csv('predictions.csv', index=False)
