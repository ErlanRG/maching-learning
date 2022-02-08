# Importing libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# Importing datasets
dataset = pd.read_csv('~/Machine_Learning_course/course/Datasets/Part 1 - Data Preprocessing/Data/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Original")
print(x)
print(y)

# Taking care of missing data
# One technique is to replace the missing data with the average of the column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])                          # Replace columns with only numerical values
x[:, 1:3] = imputer.transform(x[:, 1:3])        # This will perfom the replacement of the missing values and updates the values

# NOTE: fit() returns the mean and the standard deviation of each one of the
# features. transform() returns the result of the standarisation formula:
# Xstand = (x - mean(x))/standard deviation(x)

print("Taking care of missing data")
print(x)
print(y)

# TODO: check these lectures again to understand different methods

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] ,
                       remainder='passthrough') 
x = np.array(ct.fit_transform(x))

print("Encoding categorical data")
print (x)

# Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print("Encoding the dependent variable")
print(y)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print("Splitting the dataset")
print (x_train)
print (x_test)
print (y_train)
print (y_test)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print("Feature scaling")
print(x_train)
print(x_test)
