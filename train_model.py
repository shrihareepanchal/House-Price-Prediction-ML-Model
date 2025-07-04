# Import required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import joblib

# Load Dataset
df = pd.read_csv('train.csv')

# handle missing data
features = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','SalePrice']
df = df[features]
df = df.dropna()


# Spliting data into Train and Test Data
X = df.drop('SalePrice',axis = 1)
y = df['SalePrice']


# train test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# training Random Forest Model
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

# Calculating Model Performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)

# Saving Trained Model
joblib.dump(model,'house_price_model.pkl')
