import numpy as np
import matplotlib.pyplot as plt

"""
    Use for linearly seperable datas

    L(w,b) = Sum max(0,1 - y_n*(w.T*x_n + b)) + lambda/2 * norm_2(w) ^ 2

    ∂L(w,b)
    ------- = -Sum(y_n * x_n) + lambda * w
      ∂w

    ∂L(w,b)
    ------- = -Sum(y_n)
      ∂b
"""
np.random.seed(21)
means = [[2, 2], [4, 1]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X1[-1, :] = [2.7, 2]
X = np.concatenate((X0.T, X1.T), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # 1x20

w = np.random.randn(X.shape[0],1)
b = np.random.randn(1)

lambd = 1 / 100

def cost(w,b):
    u = y*(np.dot(w.T,X) + b)
    cost = np.sum(np.maximum(0, 1 - u)) + lambd * 0.5 * np.sum(w*w)
    return cost

def gradient(w,b):
    u = y*(np.dot(w.T,X) + b)
    less_one = np.where(u < 1)[1]

    grad_w = -np.sum(y[:,less_one]*X[:,less_one],axis=1,keepdims=True) + lambd * w
    grad_b = -np.sum(y[:,less_one])

    return (grad_w,grad_b)

def numerical(w,b):
    grad_w = np.zeros_like(w)
    grad_b = np.zeros_like(b)
    eps = 1e-6
    for i in range(len(grad_w)):
        b_left = b.copy()
        b_right = b.copy()
        b_left += eps
        b_right -= eps
        grad_b = (cost(w,b_left) - cost(w,b_right)) / (2*eps)

        w_left = w.copy()
        w_right = w.copy()
        w_left[i] += eps
        w_right[i] -= eps
        grad_w[i] = (cost(w_left,b) - cost(w_right,b)) / (2*eps)
    return (grad_w,grad_b)


def grad_descent(w0,b0,epochs,learn_rate):
    w = w0
    b = b0

    for i in range(epochs):
        grad_w,grad_b = gradient(w,b)

        w -= learn_rate*grad_w
        b -= learn_rate*grad_b

        if i % 100 == 0:
            # print('grad_w norm : %f' %np.linalg.norm(grad_w))
            # print('grad_b norm : %f' %np.linalg.norm(grad_b))
            print('iter %d' %i + ' cost: %f' %cost(w,b))
        if np.linalg.norm(grad_w) < 1e-5 and np.linalg.norm(grad_b) < 1e-5:
            break

    return (w,b)

learn_rate = 0.01
epochs = 1000000


if __name__ == '__main__':
    grad_w,grad_b = gradient(w,b)
    numgrad_w,numgrad_b = numerical(w,b)
    print(grad_w)
    print(grad_b)
    dif_w = np.linalg.norm(grad_w - numgrad_w)
    dif_b = np.linalg.norm(grad_b - numgrad_b)

    print(dif_w)
    print(dif_b)
    w,b = grad_descent(w,b,epochs,learn_rate)
    print(w)
    print(b)
    x0 = np.linspace(0,8,2)
    y0 = w[0][0]*x0 + w[1][0]*x0 + b

    print(x0)
    print(y0)
    plt.axis([1,4,-1,3])
    plt.plot(X0[:,0],X0[:,1],'ro')
    plt.plot(X1[:,0],X1[:,1],'bo')
    plt.plot(x0,y0)
    plt.show()
