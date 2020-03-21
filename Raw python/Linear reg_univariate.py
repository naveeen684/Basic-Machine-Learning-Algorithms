# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

class LinearRegressor:
    def __init__(self,x,y,alpha=0.01,b0=0,b1=0):
        self.i=0
        self.x=x
        self.y=y
        self.alpha=alpha
        self.b0=b0
        self.b1=b1
    def predict(model,x):
        return model.b0 + model.b1*x
    def derivative(model,i):
        return sum([(model.predict(x1)-y1)*1 
                    if i==0 else (model.predict(x1)-y1)*x1 
                    for x1,y1 in zip(model.x,model.y)])/len(model.x)
    def update_co(model,i):
        if i==0:
            model.b0-=model.alpha*model.derivative(i)
        elif i==1:
            model.b1-=model.alpha*model.derivative(i)
    def fit(model):
        model.i=0
        for model.i in range(1000):
            model.update_co(0)
            model.update_co(1)
    def error(model,x,y):
        return sum([(model.predict(x1)-y1)**2 
                   for x1,y1 in zip(model.x,model.y)])/2*len(x)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as t1
x=[i for i in range(12)]
y=[2*i+3 for i in range(12)]
x_train,x_test,y_train,y_test=t1(x,y,test_size=1/3,random_state=None)
linearregressor=LinearRegressor(x_train,y_train,alpha=0.037)
linearregressor.fit()
y_testpred=[linearregressor.predict(i) for i in x_test]
print(linearregressor.b1,linearregressor.b0)
print(y_testpred,y_test)
y_pred=[linearregressor.predict(i) for i in x]
print('accuracy=',(1-abs(linearregressor.error(x_test,y_test)))*100)
#actual
plt.scatter(x,y,color='black')
plt.plot(x,y,color='blue')
plt.plot(x,y_pred,color='red')
plt.title('total')      
plt.xlabel('x')
plt.ylabel('y')
plt.show()