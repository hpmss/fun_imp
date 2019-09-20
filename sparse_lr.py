import numpy as np

np.random.seed(12456)

I = 20
dim = 2
x = np.random.randn(dim,I)
x = np.concatenate((np.ones((1,I)),x),axis=0)
w = np.random.randn(I,1)

v = 10e-3

H = np.eye(dim + 1)

variance_range = np.linspace(13,1000,1500)

def inference():
    min_variance = variance_range[0]
    val = -np.log(np.random.multivariate_normal(np.zeros(I),np.dot(np.dot(x.T,np.linalg.inv(H)),x) + variance_range[0] * np.eye(I)))
    val = np.mean(val[~np.isnan(val)],axis=0)
    for i in range(1,len(variance_range)):
        var = variance_range[i]
        new_val = -np.log(np.random.multivariate_normal(np.zeros(I),np.dot(np.dot(x.T,np.linalg.inv(H)),x) + var * np.eye(I)))
        new_val = np.mean(new_val[~np.isnan(new_val)],axis=0)
        if new_val < val:
            val = new_val
            min_variance = var
    cov = min_variance * np.linalg.inv((np.dot(x,x.T) + H))
    muy = np.dot(np.dot(cov,x),w)/min_variance
    for i in range(2,dim+1):
        H[i,i] = (1 - H[i,i] * E[i,i] + v) / (muy[i]**2 + v)
        
inference()

