import numpy as np
import pandas as pd
import xgboost
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
from collections import OrderedDict

train_dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")

categorical_features=['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',
                      'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                      'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',
                      'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',
                      'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',
                      'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',
                      'MiscFeature','SaleType','SaleCondition']
every_column_except_y = [col for col in train_dataset.columns if col not in ['SalePrice','Id']]

numeric_feats = train_dataset[every_column_except_y].dtypes[train_dataset.dtypes != "object"].index
train_dataset[numeric_feats] = np.log1p(train_dataset[numeric_feats])

numeric_feats = test_dataset[every_column_except_y].dtypes[test_dataset.dtypes != "object"].index
test_dataset[numeric_feats] = np.log1p(test_dataset[numeric_feats])

features_with_nan=['Alley','MasVnrType','BsmtQual','BsmtQual','BsmtCond','BsmtCond','BsmtExposure',
                   'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish']

def ConverNaNToNAString(data, columnList):
    for x in columnList:       
        data[x] = data[x].astype(str)              

ConverNaNToNAString(train_dataset, features_with_nan)
ConverNaNToNAString(test_dataset, features_with_nan)

def CreateColumnPerValue(data, columnList):
    for x in columnList:
        values = pd.unique(data[x])
        for v in values:
            column_name = x + "_" + str(v)
            data[column_name] = (data[x] == v).astype(float)
        data.drop(x, axis=1, inplace=True)

CreateColumnPerValue(train_dataset, categorical_features)
CreateColumnPerValue(test_dataset, categorical_features)

model = xgboost.XGBRegressor(colsample_bytree=0.4,
                             gamma=0,                 
                             learning_rate=0.07,
                             max_depth=3,
                             min_child_weight=1.5,
                             n_estimators=10000,                                                                    
                             reg_alpha=0.75,
                             reg_lambda=0.45,
                             subsample=0.6,
                             random_state=42) 

every_column_except_y = [col for col in train_dataset.columns if col not in ['SalePrice','Id']]
model.fit(train_dataset[every_column_except_y], train_dataset['SalePrice'])

most_relevant_features = list(dict((k, v) for k, v in model.get_booster().get_fscore().items() if v >= 10).keys())
print(most_relevant_features)

plt.scatter(train_dataset.GrLivArea, train_dataset.SalePrice, c="blue", marker="s")
plt.title("GrLivArea vs SalePrice")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

plt.scatter(train_dataset.LotArea, train_dataset.SalePrice, c="blue", marker="s")
plt.title("LotArea vs SalePrice")
plt.xlabel("LotArea")
plt.ylabel("SalePrice")
plt.show()

train_dataset = train_dataset[train_dataset.GrLivArea < 8.25]
train_dataset = train_dataset[train_dataset.LotArea < 11.5]
train_dataset = train_dataset[train_dataset.SalePrice < 13]
train_dataset = train_dataset[train_dataset.SalePrice > 10.75]
train_dataset.drop("Id", axis=1, inplace=True)
train_x = train_dataset[most_relevant_features]
train_y = train_dataset['SalePrice']

best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                                      gamma=0,                 
                                      learning_rate=0.07,
                                      max_depth=3,
                                      min_child_weight=1.5,
                                      n_estimators=10000,                                                                    
                                      reg_alpha=0.75,
                                      reg_lambda=0.45,
                                      subsample=0.6,
                                      random_state=42)

best_xgb_model.fit(train_x, train_y)
test_dataset['Prediction'] = np.expm1(best_xgb_model.predict(test_dataset[most_relevant_features]))
filename = 'submission.csv'
pd.DataFrame({'Id': test_dataset.Id, 'SalePrice': test_dataset.Prediction}).to_csv(filename, index=False)

print(test_dataset['Prediction'].head())
print(test_dataset['Prediction'].count())
