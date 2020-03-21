from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
digits=load_digits()
plt.gray()
'''for i in range(5):
    plt.matshow(digits.images[i])'''
    
from sklearn.model_selection import train_test_split as t1
x_train,x_test,y_train,y_test=t1(digits.data,digits.target,test_size=0.10,random_state=0)

from sklearn.neural_network import MLPClassifier

NN=MLPClassifier(activation='logistic',solver='adam',hidden_layer_sizes=(30,200),random_state=None)
NN.fit(x_train,y_train)

y_pred=NN.predict(x_test)

print('Score:',NN.score(x_test,y_test))

print(pd.DataFrame({'Actual':y_test,'Predicted':y_pred}))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred))