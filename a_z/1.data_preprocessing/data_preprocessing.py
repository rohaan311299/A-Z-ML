# importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# importing the dataset
dataset=pd.read_csv("Data.csv") # pd.read_csv will form a dataframe
X = dataset.iloc[:, :-1].values # features or the independent variables
y = dataset.iloc[:, -1].values # dependent variables
# .values states that we want to collect all the values
# print(X)
# print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)

# Encoding categorical data

# Encoding the independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
# print(X)

# Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print("X_train=",X_train)
# print("y_train=",y_train)
# print("X_test=",X_test)
# print("y_test=",y_test)

# Feature Scaling (Feature Scaling won't be necessary for all the models)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.fit_transform(X_test[:,3:])
# print(X_train)
# print(X_test)