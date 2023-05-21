#!/usr/bin/env python3

# @Author: Huseyin Tutan 110190021

# IMPORT THE LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorama

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from colorama import Fore
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.model_selection import CVScores
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer


# IMPORT THE DATA

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# MISSING VALUES

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data.isna().sum())

print(data.columns)

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

plt.show()

# Filling the missing values

data['PoolQC'] = data['PoolQC'].fillna('NA')
data['MiscFeature'] = data['MiscFeature'].fillna('NA')
data['Alley'] = data['Alley'].fillna('NA')
data['Fence'] = data['Fence'].fillna('NA')
data['FireplaceQu'] = data['FireplaceQu'].fillna('NA')
imputer = KNNImputer(n_neighbors=5)
data['LotFrontage'] = imputer.fit_transform(data[['LotFrontage']])
data['GarageYrBlt'] = data['GarageType'].fillna('NA')
data['GarageType'] = data['GarageType'].fillna('NA')
data['GarageFinish'] = data['GarageFinish'].fillna('NA')
data['GarageQual'] = data['GarageQual'].fillna('NA')
data['GarageCond'] = data['GarageCond'].fillna('NA')
data['BsmtExposure'] = data['BsmtExposure'].fillna('NA')
data['BsmtFinType2'] = data['BsmtFinType2'].fillna('NA')
data['BsmtFinType1'] = data['BsmtFinType1'].fillna('NA')
data['BsmtCond'] = data['BsmtCond'].fillna('NA')
data['BsmtQual'] = data['BsmtQual'].fillna('NA')
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data['Electrical'] = data['Electrical'].fillna('SBrkr')

