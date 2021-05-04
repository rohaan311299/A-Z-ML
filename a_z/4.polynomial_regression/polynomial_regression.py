# Importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

# Training the linear regression model on whole dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)

# Training the polynomial regression model on whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualising the linear regression results
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the polynomial regression results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg_2.predict(X_poly),color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting the results with linear regression
print(regressor.predict([[6.5]]))

# Predicting the results with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))