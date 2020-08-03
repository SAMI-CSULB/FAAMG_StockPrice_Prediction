# Kismet USA Inc.

# Importing the libraries
import numpy as np
import pandas as pd
# [0,2,3,4,5,6,7,8,9,10,11]
# Importing the dataset
dataset = pd.read_csv('Google Intraday Edited.csv')
#Getting High Low and Open as input values
X = dataset.iloc[:, 1:4].values
#Settting CLose as Output
y = dataset.iloc[:, 0].values
y= np.reshape(y,(-1,1))

# Replacing 'NAN' in output dataset with '0'
from sklearn.impute import SimpleImputer
imputer = SimpleImputer (missing_values = np.nan, strategy ="constant",fill_value= 0)
imputer= imputer.fit(X[:, :])
X[:,:] = imputer.transform(X[:,:])
imputer = SimpleImputer (missing_values = np.nan, strategy ="constant",fill_value=0)
imputer= imputer.fit(y)
y = imputer.transform(y)

#Data Cleaning (Removing rows containing 0 for Low Open High and Close)
X = pd.DataFrame(X)
X.columns = ['High','Low','Open']
X = X[X.Open != 0]
y = pd.DataFrame(y)
y.columns = ['Close']
y = y[y.Close != 0]

# Splitting the dataset into the Training set and Test set (30% used for training and 70% for testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 0,shuffle=True)


# Fitting Mutliple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting "Close" values using Test Set
y_pred = regressor.predict(X_test)
y_test['Close'] = y_test['Close'].astype(float)

###############################################################################

from sklearn.tree import DecisionTreeRegressor
regressorDT = DecisionTreeRegressor(random_state = 0, criterion = 'friedman_mse')
regressorDT.fit(X_train, y_train)
# Predicting a new result 
y_pred_DT = regressorDT.predict(X_test)


###############################################################################

# Fitting SVR to the data set
from sklearn.svm import SVR
regressorSVR = SVR(kernel='callable')
regressorSVR.fit(X_train, y_train)
# Predicting a new result 
y_pred_SVR = regressorSVR.predict(X_test)

###############################################################################

# Fitting the Regression Model to the data set
from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators = 300, random_state=0)
regressorRF.fit(X_train, y_train)
# Predicting a new result 
y_pred_RF = regressorRF.predict(X_test)




#Concatenating Predicted and Actual "Close" values to visualize differences

result = np.concatenate((y_pred.reshape(len(y_pred),1),y_test),1)
result = pd.DataFrame(result)
result.columns = ['Predicted Close Value', 'Actual Close Value']


resultDT = np.concatenate((y_pred_DT.reshape(len(y_pred_DT),1),y_test),1)
resultDT = pd.DataFrame(resultDT)
resultDT.columns = ['Predicted Close Value using DT', 'Actual Close Value']


resultSVR = np.concatenate((y_pred_SVR.reshape(len(y_pred_SVR),1),y_test),1)
resultSVR = pd.DataFrame(resultSVR)
resultSVR.columns = ['Predicted Close Value using DT', 'Actual Close Value']

resultRF = np.concatenate((y_pred_RF.reshape(len(y_pred_RF),1),y_test),1)
resultRF = pd.DataFrame(resultRF)
resultRF.columns = ['Predicted Close Value using DT', 'Actual Close Value']


resultRF.to_csv (r'C:\Users\HP\Desktop\Kismet Project 1\Intraday FAAMG\Google\Google Intraday Decision Tree state using Random Forest with 300 estimators and MSE.csv', index= True, header=True)

import requests

