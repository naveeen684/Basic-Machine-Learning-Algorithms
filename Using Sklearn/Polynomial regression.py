from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=load_boston()
boston=pd.DataFrame(dataset.data,columns=dataset.feature_names)
boston['MEDV']=dataset.target

x=pd.DataFrame(np.c_[boston['LSTAT'],boston['RM']])
y=boston['MEDV']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=None)

from sklearn.linear_model import LinearRegression
L=LinearRegression()
metrain=[]
metest=[]

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
for i in range(1,4):
    poly_reg=PolynomialFeatures(degree=i)
    x_train=poly_reg.fit_transform(x_train)
    x_test=poly_reg.fit_transform(x_test)
    L.fit(x_train,y_train)
    y_train_predict =L.predict(x_train)
    y_test_predict = L.predict(x_test)
    metrain.append(mean_squared_error(y_train, y_train_predict))
    metest.append(mean_squared_error(y_test, y_test_predict))

print(pd.DataFrame({'Train error':metrain,'test error':metest}))
plt.plot([1,2,3],metrain,color='RED')
plt.plot([1,2,3],metest,color='BLUE')
plt.show()

