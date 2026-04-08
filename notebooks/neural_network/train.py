import numpy as np

np.random.seed(42)

grad = 1.0

for i in range(10):
    grad *= 0.5
    print(f"Layer {i+1}: Gradient = {grad}")