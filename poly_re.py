import numpy as np
import matplotlib.pyplot as plt

x = np.array([[5,39,50,41,19,23,1,20,35,45,12,4,36]]).T
x_origin = x.copy()
degree = 3
for i in range(1,degree):
    x_square = x ** (i + 1)
    x = np.concatenate((x,x_square),axis=1)
x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
y = np.array([[0,5,4,9,5,3,7,1,6,10,3,4,9]]).T
print(x)
plt.xlabel("X")
plt.ylabel("Y")

def quadratic(x,w):
    w0 = w[0][0]
    w1 = w[1][0]
    w2 = w[2][0]
    a = []
    for x_ in x:
        y = w0 + w1 * x_[2] + w2 * (x_[2] ** 2)
        a.append(y)
    return a

def lin_re_derivative(x,y):
    return np.dot(np.linalg.pinv(np.dot(x.T,x)),np.dot(x.T,y))

w = lin_re_derivative(x,y)
x0 = np.linspace(1,50)
y_total = ""
for i in range(degree + 1):
    y_part = "{} * ({}**{}) + ".format(w[i][0],x0,i)
    y_total += y_part

print(y_total)

# y0 = w0 + w1 * x0 + w2 * (x0 ** 2) + w3 * (x0 ** 3)
# plt.plot(x_origin.T,y.T,'ro')
# plt.plot(x0,y0)
# plt.show()
