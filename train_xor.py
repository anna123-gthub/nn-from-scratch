"""Sanity check: XOR is not linearly separable, so a hidden layer is required."""
import numpy as np
from nn import MLP, Adam

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])
net = MLP([2, 8, 2], seed=0)
net.fit(X, y, Adam(lr=0.05), epochs=300, batch_size=4, verbose=False)
print("predictions:", net.predict(X), "target:", y)
print("accuracy:", net.accuracy(X, y))