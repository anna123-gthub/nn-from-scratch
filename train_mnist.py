"""Train the from-scratch MLP on full MNIST (28x28, 60k train / 10k test).

Needs internet the first time (scikit-learn downloads MNIST from OpenML and caches it).
The test set is only used once, at the very end.
"""
import time
import numpy as np
from sklearn.datasets import fetch_openml
from nn import MLP, Adam


def load_mnist():
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float64) / 255.0
    y = y.astype(np.int64)
    # Standard split: first 60,000 train, last 10,000 test
    return X[:60000], y[:60000], X[60000:], y[60000:]


if __name__ == "__main__":
    X, y, X_test, y_test = load_mnist()

    # Hold out 5,000 training images to watch validation accuracy
    X_tr, y_tr = X[:55000], y[:55000]
    X_val, y_val = X[55000:], y[55000:]

    net = MLP([784, 128, 10], seed=0)
    start = time.time()
    history = net.fit(X_tr, y_tr, Adam(lr=1e-3), epochs=10, batch_size=128,
                      X_val=X_val, y_val=y_val)
    print(f"training time: {time.time() - start:.1f}s")

    print(f"\nFinal test accuracy: {net.accuracy(X_test, y_test):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["loss"]); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss")
        ax[1].plot(history["val_acc"]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation accuracy")
        plt.tight_layout()
        plt.savefig("mnist_curves.png", dpi=150)
        print("saved mnist_curves.png")
    except ImportError:
        pass