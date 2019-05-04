import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
    L(w,b) = Sum max(0,1 - y_n*(w.T*x_n + b)) + lambda/2 * norm_2(w) ^ 2
"""
np.random.seed(21)
means = [[2, 2], [4, 1]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X1[-1, :] = [2.7, 2]
X = np.concatenate((X0.T, X1.T), axis = 1)
Y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # 1x20

w = np.random.randn(X.shape[0],1)
b = np.random.randn(1)

w = tf.convert_to_tensor(w)
b = tf.convert_to_tensor(b)

lambd = 1 / 100
learn_rate = 0.01
epochs = 10000

x = tf.placeholder('float64')
y = tf.placeholder('float64')
w = tf.Variable(w)
b = tf.Variable(b)

u = y*(tf.matmul(tf.transpose(w),x) + b)
cost = tf.reduce_sum(tf.maximum(tf.constant(0,dtype='float64'),1 - u)) + lambd * 0.5 * tf.reduce_sum(w*w)

train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(epochs):
        _,cost_val = session.run([train_op,cost],feed_dict={x:X,y:Y})
        if i % 100 == 0:
            print('iter %d' %i + ' cost: %f' %cost_val)

    w = session.run(w)
    b = session.run(b)
    print(w)
    print(b)

    x0 = np.linspace(0,8,2)
    y0 = w[0][0]*x0 + w[1][0]*x0 + b[0]

    plt.axis([1,4,-1,3])
    plt.plot(X0[:,0],X0[:,1],'ro')
    plt.plot(X1[:,0],X1[:,1],'bo')
    plt.plot(x0,y0)
    plt.show()
