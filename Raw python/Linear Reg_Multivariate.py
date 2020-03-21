class LinearRegressor:
    def __init__(self,x1,x2,y,alpha=0.001,b0=0,b1=0,b2=0):
        self.x1=x1
        self.x2=x2
        self.y=y
        self.alpha=alpha
        self.b0=b0
        self.b1=b1
        self.b2=b2
    def predict(model,x1,x2):
        return model.b0+model.b1*x1+model.b2*x2
    def derivative(model,i):
        return sum([(model.predict(x1,x2)-y1)*1 if i==0 else
                   (model.predict(x1,x2)-y1)*x1 if i==1 else
                   (model.predict(x1,x2)-y1)*x2 
                   for x1,x2,y1 in zip(model.x1,model.x2,model.y)])/len(model.x1)
    def update_co(model,i):
        if i==0:
            model.b0-=model.alpha*model.derivative(i)
        elif i==1:
            model.b1-=model.alpha*model.derivative(i)
        elif i==2:
            model.b2-=model.alpha*model.derivative(i)
    def fit(model):
        model.i=0
        for model.i in range(1000):
            model.update_co(0)
            model.update_co(1)
            model.update_co(2)
            
import matplotlib.pyplot as plt
x1=[i for i in range(12)]
x2=[2*i+1 for i in range(12)]
y=[2*x+y1 for x,y1 in zip(x1,x2)]
x1_train=x1[:7]
x2_train=x2[:7]
y_train=y[:7]
x1_test=x1[7:]
x2_test=x2[7:]
y_test=y[7:]
linearregressor=LinearRegressor(x1_train,x2_train,y_train,alpha=0.01)
linearregressor.fit()
y_testpred=[linearregressor.predict(i,j) for i,j in zip(x1_test,x2_test)]
print(y_testpred,y_test)
y_pred=[linearregressor.predict(i,j) for i,j in zip(x1,x2)]
#actual
from mpl_toolkits.mplot3d import Axes3D
plt.scatter(x1,x2,y,color='black')
plt.plot(x1,x2,y_pred,color='red')
plt.title('total')      
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt1=plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt1.figure()
ax=fig.gca(projection='3d')

ax.scatter(x1,x2,y,color='black')
ax.plot(x1,x2,y_pred,color='red')
plt1.show()