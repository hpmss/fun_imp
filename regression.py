import numpy as np
import matplotlib.pyplot as plt

x = np.array([[20,14,30,56,47,95,10,36,20,11,66,44]])
x_trans = x.T
b = np.ones((x_trans.shape[0],1))

X = np.concatenate((x_trans,b),axis = 1) # N x 2
Y_Label = np.array([[14,8,20,38,34,80,5,23,14,6,43,32]]).T
W = np.random.randn(x_trans.shape[0],2)

#Linear: y_pred = w.T * x
#Loss : L = 1/2(Y_label - Y_pred)^2

derived_W_trans = np.linalg.pinv(X).dot(np.linalg.pinv(X.T)).dot(np.dot(X.T,Y_Label))

w_0 = derived_W_trans[0][0]
w_1 = derived_W_trans[1][0]

x0 = np.linspace(10,100,2)
y0 = w_1 + w_0*x0

plt.plot(x,Y_Label.T,'ro')
plt.plot(x0,y0)
plt.axis([0,100,1,90])
plt.xlabel('Money Put In')
plt.ylabel('Percent of winning')
plt.show()
