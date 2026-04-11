import numpy as np

def dropout(a, rate=0.5):
    mask = np.random.rand(*a.shape) > rate
    return a * mask
    
z1 = np.dot(X, W1) + b1
a1 = relu(z1)

a1 = dropout(a1, rate=0.5)