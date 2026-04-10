import numpy as np

lambda_reg = 0.01

l2_penalty = lambda_reg * np.sum(W1**2)

loss = original_loss + l2_penalty