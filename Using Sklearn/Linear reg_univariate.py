import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv(r'C:\Users\elcot\Desktop\naveen\Datasets\student_scores.csv')
print(dataset)
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split as T
x_train,x_test,y_train,y_test=T(x,y,test_size=1/3,random_state=0)
print(x_train)

from sklearn.linear_model import LinearRegression 
LR=LinearRegression()
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
print(LR.intercept_)
print(LR.coef_)
from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_pred),metrics.mean_squared_error(y_test,y_pred),
np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df)
plt.scatter(x,y,color='black')
plt.plot(x_test,y_pred,color='red')
plt.title('Students study relation')
plt.xlabel('hours')
plt.ylabel('percentage')
plt.show()
