
class LinearRegressor:
    def __init__(self,x,y,alpha=0.01,b0=0,b1=0,b2=0):
        self.i=0
        self.x=x
        self.y=y
        self.alpha=alpha
        self.b0=b0
        self.b1=b1
        self.b2=b2
        self.lam=80
    def predict(model,x):
        return model.b0 + model.b1*x+model.b2*x*x
    def derivative(model,i):
        return sum([(model.predict(x1)-y1)*1 
                    if i==0 else (model.predict(x1)-y1)*x1 if i==1
                    else (model.predict(x1)-y1)*x1*2
                    for x1,y1 in zip(model.x,model.y)])/len(model.x)
    def update_co(model,i):
        if i==0:
            model.b0-=model.alpha*(model.derivative(i)+(model.lam*2*model.b0)/len(model.x))
        elif i==1:
            model.b1-=model.alpha*(model.derivative(i)+(model.lam*2*model.b1)/len(model.x))
        elif i==2:
            model.b2-=model.alpha*(model.derivative(i)+(model.lam*2*model.b2)/len(model.x))
    def fit(model):
        model.i=0
        for model.i in range(1000):
            model.update_co(0)
            model.update_co(1)
            model.update_co(2)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as t1
x=[i for i in range(12)]
y=[0,4,15.5,36,65,121,144,150,256,324,480,484]
x_train,x_test,y_train,y_test=t1(x,y,test_size=1/3,random_state=None)
linearregressor=LinearRegressor(x_train,y_train,alpha=0.001)
linearregressor.fit()
y_testpred=[linearregressor.predict(i) for i in x_test]
df=pd.DataFrame({'Actual':y_test,'Predicted':y_testpred})
print(df)
y_pred=[linearregressor.predict(i) for i in x]
#actual
plt.scatter(x,y,color='black')
plt.plot(x,y_pred,color='red')
plt.title('total')      
plt.xlabel('x')
plt.ylabel('y')
plt.show()