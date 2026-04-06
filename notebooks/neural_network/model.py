# model.py

import numpy as np

class SimpleNN:

    def __init__(self):
        np.random.seed(42)
        self.W1 = np.random.randn(2, 2)
        self.b1 = np.zeros(2)
        self.W2 = np.random.randn(2, 1)
        self.b2 = 0

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2
        
    def backward(self, X, y, learning_rate=0.01):

        dz2 = self.a2.flatten() - y
        dW2 = np.dot(self.a1.T, dz2.reshape(-1,1))
        db2 = np.mean(dz2)

        da1 = np.dot(dz2.reshape(-1,1), self.W2.T)
        dz1 = da1 * (self.z1 > 0)

        dW1 = np.dot(X.T, dz1)
        db1 = np.mean(dz1, axis=0)

        # update
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1