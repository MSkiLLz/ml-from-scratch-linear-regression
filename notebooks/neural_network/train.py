import numpy as np

np.random.seed(42)

W1 = np.random.randn(2, 3)
b1 = np.zeros(3)

W2 = np.random.randn(3, 2)
b2 = np.zeros(2)

W3 = np.random.randn(2, 1)
b3 = 0

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)

    return z1, a1, z2, a2, z3, a3
    
X = np.array([[1,2],[2,3]])

_, _, _, _, _, output = forward(X)

print(output)