import numpy as np
import matplotlib.pyplot as plt

"""
    This is an naive implementation of DLT algorithm. 
    Given n >= 4 2D to 2D point correspondences {xi <-> xi'} , determine the 2D homography matrix H such that xi' = Hxi.

    In order to have H such that xi' = Hxi. It is required that the cross_product(xi',Hxi) = 0 or xi' 'cross' Hxi = 0.
    It can be rewrited as Ah = 0, where h is the collumn vector with entries correspond to entries in H.

    However , one cannot determine the exact solution for h apart from the 0 vector. So it is possible to attempt to minimize the norm |Ah| , subject to |h| = 1.
    Which means h can be chosen to be the (unit) eigenvector with the least eigenvalue of A.T*A. Since minimizing |Ah| is the same as minimizing |Ah|^2.
    |Ah|^2 = h.T * (A.T*A) * h , this makes A.T*A symmetric which can be orthogonally diagonalized.
    Hence we can use SVD to minimize |Ah|.
"""

np.random.seed(54618)

x_nor = np.random.randn(2,5)
x_nor = np.concatenate((x_nor,np.ones((1,5))),axis=0)

x_nor_copy = x_nor.copy()

x_prime = np.random.randn(2,5)
x_prime = np.concatenate((x_prime,np.ones((1,5))),axis=0)

x_prime_copy = x_prime.copy()

A = np.zeros((10,9))

def dlt_unnormalized():
    #Construct matrix A
    for i in range(5):
        x = np.array([x_nor[:,i]]).T
        x_pr = np.array([x_prime[:,i]]).T

        w_i_pr_x_i_pos = x_pr[-1,0] * x.T
        w_i_pr_x_i_neg = -w_i_pr_x_i_pos
        y_i_pr_x_i = x_pr[1,0] * x.T
        x_i_pr_x_i = -x_pr[0,0] * x.T
        
        k = 2 * i

        A[k,3:6] = w_i_pr_x_i_neg[0,:]
        A[k,6:9] = y_i_pr_x_i[0,:]
        A[k+1,0:3] = w_i_pr_x_i_pos[0,:]
        A[k+1,6:9] = x_i_pr_x_i[0,:]

    #Decompose A = U*D*V.T
    svd = np.linalg.svd(A,full_matrices=False)
    v = svd[-1].T
    h = v[:,-1]

    H = np.zeros((3,3))

    H[0,:] = h[0:3]
    H[1,:] = h[3:6]
    H[2,:] = h[6:9]

    x_prime_H = np.dot(H,x_nor)

    for i in range(5):
        x_prime_H[:,i] = x_prime_H[:,i] / x_prime_H[:,i][-1]

    print(x_prime_H)
    print(x_prime)

    fig = plt.figure(figsize=(4,4))
    fig.add_subplot(111)
    plt.plot(x_nor[0:2,:][0],x_nor[0:2,:][1],'bo')
    plt.plot(x_prime[0:2,:][0],x_prime[0:2,:][1],'ro')
    plt.plot(x_prime_H[0:2,:][0],x_prime_H[0:2,:][1],'go')
    plt.show()

dlt_unnormalized()

def dlt_normalized():

    #x normalization
    global x_nor,x_prime
    x_nor_centroid = np.array([np.sum(x_nor[0:3,:],axis=1) / x_nor.shape[1]]).T
    tx,ty = -x_nor_centroid[0,0],-x_nor_centroid[1,0]
    sum = 0
    for i in range(x_nor.shape[1]):
        sum += np.sqrt((x_nor[0,i] + tx)**2 + (x_nor[1,i] + ty)**2)
    s = x_nor.shape[1]*np.sqrt(2) / sum
    T = np.array([[s,0,s*tx],[0,s,s*ty],[0,0,1]])
    x_nor = np.dot(T,x_nor)

    #x_prime normalization same as above
    x_prime_centroid = np.array([np.sum(x_prime[0:3,:],axis=1) / x_prime.shape[1]]).T
    tx,ty = -x_prime_centroid[0,0],-x_prime_centroid[1,0]
    sum = 0
    for i in range(x_prime.shape[1]):
        sum += np.sqrt((x_prime[0,i] + tx)**2 + (x_prime[1,i] + ty)**2)
    s = x_prime.shape[1]*np.sqrt(2) / sum
    T_prime = np.array([[s,0,s*tx],[0,s,s*ty],[0,0,1]])
    x_prime = np.dot(T_prime,x_prime)

    #Print this to verify that average distance is sqrt(2) and transformed centroid is at origin
    avg_d = 0
    for i in range(5):
        avg_d += np.sqrt(np.sum(x_prime[0:2,i]**2))
    avg_d /= 5
    print(avg_d)
    print(np.array([np.sum(x_prime[0:3,:],axis=1) / x_prime.shape[1]]).T)

    print(x_nor_copy)
    print(x_prime_copy)
    for i in range(5):
        x = np.array([x_nor[:,i]]).T
        x_pr = np.array([x_prime[:,i]]).T

        w_i_pr_x_i_pos = x_pr[-1,0] * x.T
        w_i_pr_x_i_neg = -w_i_pr_x_i_pos
        y_i_pr_x_i = x_pr[1,0] * x.T
        x_i_pr_x_i = -x_pr[0,0] * x.T
        
        k = 2 * i

        A[k,3:6] = w_i_pr_x_i_neg[0,:]
        A[k,6:9] = y_i_pr_x_i[0,:]
        A[k+1,0:3] = w_i_pr_x_i_pos[0,:]
        A[k+1,6:9] = x_i_pr_x_i[0,:]

    svd = np.linalg.svd(A,full_matrices=False)
    v = svd[-1].T
    h = v[:,-1]

    H_transformed = np.zeros((3,3))

    H_transformed[0,:] = h[0:3]
    H_transformed[1,:] = h[3:6]
    H_transformed[2,:] = h[6:9]


    #Denormalization
    H_true = np.dot(np.dot(np.linalg.inv(T_prime),H_transformed),T)
    x_prime_H = np.dot(H_true,x_nor_copy)
    for i in range(5):
        x_prime_H[:,i] = x_prime_H[:,i] / x_prime_H[:,i][-1]
    print(x_prime_H)

    fig = plt.figure(figsize=(4,4))
    fig.add_subplot(111)
    plt.plot(x_nor_copy[0:2,:][0],x_nor_copy[0:2,:][1],'bo')
    plt.plot(x_prime_copy[0:2,:][0],x_prime_copy[0:2,:][1],'ro')
    plt.plot(x_prime_H[0:2,:][0],x_prime_H[0:2,:][1],'go')
    plt.show()


dlt_normalized()
