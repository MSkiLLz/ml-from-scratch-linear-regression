import numpy as np

def batch_norm(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-8  # avoid division by zero
    return (x - mean) / std
    
z1 = np.dot(X, W1) + b1
z1_norm = batch_norm(z1)

a1 = relu(z1_norm)