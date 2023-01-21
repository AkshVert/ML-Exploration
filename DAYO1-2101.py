# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 17:25:46 2023

@author: Akshay

"""

# Library Importing 

#Pandas(Python library) used as alias pd 
#Usage of Pandas Library - It helps in data managing
import pandas as pd
#Matplotlip used as alias plt 
#mathploblib.pyplot is a collection of functions used to create a figure, plotting in a figure, axis and for decorating the plot, etc..
#import matplotlib as pltt
#import matplotlib.pyplot as plt
#Seaborn used as alias sns 
#seaborn is also a library that uses Matplotlib to plot grphs and visualize random distributions.
#import seaborn as sns

# Print Library Versions
#print("Version to know of library [%s]"%s(<lib_name>.__version__))
print("Version of pandas is [%s]"%(pd.__version__))
#print("Version of matplotlib is [%s]"%(pltt.__version__))
#print("Version of seaborn is [%s]"%(sns.__version__))

# Reading DataSet
# Dataset is a collection of data used to train the model.
#.read_excel ----> ?
# Input - excel file 
# Output - 
dataset = pd.read_excel("HousePricePrediction.xlsx")
print(type(dataset))
#Printing the dataset
#dataset.shape is used to determine no. of rows and colums in the dataset
print(dataset.shape)

# Dataset Cleaning
# drop() removes the specified row or column ('Id' in this case)
#
dataset.drop(['Id'],
             #axis=1 is used for column while axis=0 for the row
             axis=1,
             inplace=True)


print(dataset.shape)
# fillna replaces the missing value with user specified value 
# Here missing value is replaced with mean values of the SalePrice
dataset['SalePrice'] = dataset['SalePrice'].fillna(
dataset['SalePrice'].mean())

#dropna  removes the missing values(rows/columns)
new_dataset = dataset.dropna()
print(dataset.shape)
new_dataset.isnull().sum()



from sklearn.preprocessing import OneHotEncoder
#OneHotEncoder is used to convert categorical data into numerical data
 
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
#df_final.drop to drop the data from (column-Saleprice, column),
X = df_final.drop(['SalePrice'], axis=1)
# 
Y = df_final['SalePrice']

# Split the training set into
# training and validation set
# Training part of the sequence 
X_train, X_valid, Y_train, Y_valid = train_test_split(
	X, Y, train_size=0.8, test_size=0.2, random_state=0)

# svm performs supervised learning for classification and regression of the data groups.
from sklearn import svm
# svc used for classification of the tasks.
from sklearn.svm import SVC
# mean_absolute_percentage_error(MAPE) is mean of absolute percentage error
from sklearn.metrics import mean_absolute_percentage_error
 
model_SVR = svm.SVR()

model_SVR.fit(X_train,Y_train)

Y_pred = model_SVR.predict(X_valid)
 
print(Y_pred)

