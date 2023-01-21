# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:38:42 2023

@author: Akshay
"""

# Library Importing 

import pandas as pd
import matplotlib as pltt
import matplotlib.pyplot as plt
import seaborn as sns

# Print Library Versions

print("Version of pandas is [%s]"%(pd.__version__))
print("Version of matplotlib is [%s]"%(pltt.__version__))
print("Version of seaborn is [%s]"%(sns.__version__))

# Reading DataSet
dataset = pd.read_excel("HousePricePrediction.xlsx")
# Printing DataSet

print(dataset.shape)

# Dataset Cleaning
dataset.drop(['Id'],
             axis=1,
             inplace=True)

print(dataset.shape)
dataset['SalePrice'] = dataset['SalePrice'].fillna(
dataset['SalePrice'].mean())

new_dataset = dataset.dropna()
print(dataset.shape)
new_dataset.isnull().sum()

from sklearn.preprocessing import OneHotEncoder
 
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ',
      len(object_cols))



OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
	X, Y, train_size=0.8, test_size=0.2, random_state=0)


from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
 
model_SVR = svm.SVR()

model_SVR.fit(X_train,Y_train)

Y_pred = model_SVR.predict(X_valid)
 
print(Y_pred)


