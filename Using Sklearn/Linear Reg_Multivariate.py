import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('D:\petrol_consumption.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split as T
x_train,x_test,y_train,y_test=T(x,y,test_size=1/3,random_state=None)

from sklearn.linear_model import LinearRegression 
LR=LinearRegression()
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
print(LR.intercept_)
print(LR.coef_)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df)
df.to_csv('Naveen.csv')

