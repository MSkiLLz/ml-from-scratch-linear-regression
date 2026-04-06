# train.py

from model import SimpleNN
import numpy as np

X = np.array([[1,2],[2,3],[3,4],[5,6]])
y = np.array([0,0,1,1])

model = SimpleNN()

for epoch in range(100):

    preds = model.forward(X)

    loss = -np.mean(y*np.log(preds.flatten()) +
                    (1-y)*np.log(1-preds.flatten()))

    model.backward(X, y)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print(model.forward(X))