# importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# importing the dataset
dataset=pd.read_csv("./Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# splitting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)

# training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred=regressor.predict(X_test)
print(y_pred)

# visualising the training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs experience (Training Set)")
plt.xlabel("Experience(years)")
plt.ylabel("Salary")
plt.show()

# visualising the test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs experience (Training Set)")
plt.xlabel("Experience(years)")
plt.ylabel("Salary")
plt.show()