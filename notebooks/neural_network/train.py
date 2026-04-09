import numpy as np

w = 5
gradient = 2

for lr in [0.01, 0.1, 1]:
    new_w = w - lr * gradient
    print(f"LR={lr}, Updated w={new_w}")