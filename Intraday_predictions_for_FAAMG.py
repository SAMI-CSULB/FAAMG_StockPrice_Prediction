# Kismet USA Inc.

# Importing the libraries
import numpy as np
import pandas as pd
# [0,2,3,4,5,6,7,8,9,10,11]
# Importing the dataset
dataset = pd.read_csv('Daily_Data.csv')
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

#Concatenating Predicted and Actual "Close" values to visualize differences

result = np.concatenate((y_pred.reshape(len(y_pred),1),y_test),1)
result = pd.DataFrame(result)
result.columns = ['Predicted Close Value', 'Actual Close Value']
