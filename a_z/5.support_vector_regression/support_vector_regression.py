# Importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Import the dataset
dataset=pd.read_csv("./Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

# print(X)
# print(y)

# reshaping y to a 2d array to avoid errors when feature scalling using StandardScalar
y=y.reshape(len(y),1)
# print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

# print(X)
# print(y)

# Training the svr model on the dataset
from sklearn.svm import SVR 
regressor=SVR(kernel="rbf")
regressor.fit(X,y)

# Predicting a new result
# value which we want to use for the prediction has to be scaled on the same scale as the features
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
# print(y_pred)

# Visualising the SVR Results
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color="red")
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)),color="blue")
plt.title("Truth or Bluff (Support Vector Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising SVR results ( for higher and smoother curver)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()