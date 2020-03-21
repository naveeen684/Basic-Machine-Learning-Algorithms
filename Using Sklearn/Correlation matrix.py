from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
dataset=load_boston()
boston=pd.DataFrame(dataset.data,columns=dataset.feature_names)
boston['MEDV']=dataset.target

'''import seaborn as sns 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt1.show()


correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)'''

'''plt.subplot(1,2,1)
plt.scatter(boston['RM'],boston['MEDV'],color='BLACK') 
plt.ylabel('MEDV')
plt.xlabel(boston['RM'])'''


'''plt.subplot(1,2,2)
plt.scatter(boston['LSTAT'],boston['MEDV'],color='RED') 
plt.ylabel('MEDV')
plt.xlabel('LSTAT')'''

import numpy as np

x=pd.DataFrame(np.c_[boston['LSTAT'],boston['RM']])
y=boston['MEDV']
print(x,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=None)

from sklearn.linear_model import LinearRegression

L=LinearRegression()
L.fit(x_train,y_train)
metrain=[]
metest=[]
y_train_predict =L.predict(x_train)

from sklearn.metrics import mean_squared_error
metrain.append(mean_squared_error(y_train, y_train_predict))

print("The model performance for training set")
print('RMSE is ',metrain)
print("\n")

y_test_predict = L.predict(x_test)
metest.append(mean_squared_error(y_test, y_test_predict))

print("The model performance for testing set")
print('RMSE is ',metest)
print("\n")

print(pd.DataFrame({'Actual':y_test,'Predicted':y_test_predict}))

