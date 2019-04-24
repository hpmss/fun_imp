import numpy as np
import matplotlib.pyplot as plt


"""
    @hpms - 22/04/2019

    y = x_N^4 * b4 + x_N^3 * b3 + x_N^2 * b2 + x_N*b1 + b0 (degree 4)

    Where 'x_N' is a sample and 'b' is a coefficient of the corresponding degree

    => Extends to vector form : [x^4,x^3,x^2,x^1,1] * [b4,b3,b2,b1,b0]

    => Extends to matrix form: [x_1^4,x_1^3,x_1^2,x_1^1,1]  [b4]    =   [y1]
                               [x_2^4,x_2^3,x_2^2,x_2^1,1]  [b3]    =   [y2]
                               [x_3^4,x_3^3,x_3^2,x_3^1,1]  [b2]    =   [y3]
                               [x_4^4,x_4^3,x_4^2,x_4^1,1]  [b1]    =   [y4]
                               ...........................  [b0]    =   ....
                               [x_N^4,x_N^3,x_N^2,x_N^1,1]          =   [yN]
"""

DEGREE = 5
SIZE = 50
X_flat = np.random.randn(SIZE) # Size x 1
X = X_flat.copy().reshape(SIZE,1)

Y = np.random.randn(SIZE,1)

X = np.concatenate((np.ones((SIZE,1)),X),axis=1)
for i in range(2,DEGREE + 1):
    X_power_i = X_flat ** i
    X_power_i = X_power_i.reshape(SIZE,1)
    X = np.concatenate((X,X_power_i),axis = 1)

""" Solve for W using least-square => W'

    X*W = y
    => It is possible for W to have no solution
    => Using least-square theorem : X*W' = y'
    => Find W' such that X*W' = y' , y' is close to y

    argmin || y - y' ||

    => Find a vector y' that lies in col_space(X) and has min || y - y' ||
    => Project y onto X = y'
    => X*W' = proj_y_col(X)

    => X*W' - y = proj_y_col(X) - y
    => { Comp_ortho(A) : X*W' - y }
    => {Null(A.T) : X*W' - y}

    => A.T*(X*W' - y) = 0

    => W' = (X.T * X)^-1 * X.T * Y


"""
W = np.linalg.pinv(np.dot(X.T,X)).dot(np.dot(X.T,Y))

W_coeff = W[:,0]

X0 = np.linspace(-4,4,1000)
Y0 = W_coeff[0]

for i in range(1,len(W_coeff)):
    w_i = W_coeff[i]
    Y0 = Y0 + w_i * (X0 ** i)

plt.axis([-7,7,-7,7])
plt.plot(X_flat,Y[:,0],'ro')
plt.plot(X0,Y0)
plt.xlabel("Guassian Distribution")
plt.ylabel("Polynomial Regression Of Degree %d" % (DEGREE))
plt.show()
