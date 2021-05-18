# Importing the libraries
import numpy as np 
import pandas as pd 
import tensorflow as tf 

# Importing the dataset
dataset=pd.read_csv("./Churn_Modelling.csv")
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

# Encoding categorical Data

# lable encoding the "Gender" Column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(x[:,2])

# One hot encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X=np.array(ct.fit_transform(X))

# Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.prep import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

# Building the ANN

# initializing the ann
ann=tf.keras.model.Sequential()

# adding the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Training the ANN

# compiling the ann
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# training the ann on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# predicting the results of a single observation
print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))>0.5)

# predicting test set results
y_pred=ann.predict(X_test)
y_pred=(y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)