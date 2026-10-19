"""Neural network from scratch in NumPy."""
import numpy as np


# ---------- 1. Layers ----------
class Dense:
    """y = x @ W + b"""

    def __init__(self, n_in, n_out, rng):
        # He initialization: good default for ReLU
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)

    def forward(self, x):
        self.x = x  # cache for backward
        return x @ self.W + self.b

    def backward(self, grad_out):
        # grad_out = dLoss/dy, shape (batch, n_out)
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T  # dLoss/dx, passed to previous layer

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask

    def params(self):
        return []


# ---------- 2. Loss ----------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Combines softmax + cross-entropy; the gradient simplifies to (p - y)/N."""

    def forward(self, logits, y):
        self.p = softmax(logits)
        self.y = y
        n = len(y)
        return -np.log(self.p[np.arange(n), y] + 1e-12).mean()

    def backward(self):
        n = len(self.y)
        grad = self.p.copy()
        grad[np.arange(n), self.y] -= 1
        return grad / n


# ---------- 4. Optimizers ----------
class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params):
        for p, g in params:
            p -= self.lr * g


class Momentum:
    def __init__(self, lr=0.05, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, {}

    def step(self, params):
        for p, g in params:
            v = self.v.setdefault(id(p), np.zeros_like(p))
            v *= self.beta
            v += g
            p -= self.lr * v


class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params):
        self.t += 1
        for p, g in params:
            m = self.m.setdefault(id(p), np.zeros_like(p))
            v = self.v.setdefault(id(p), np.zeros_like(p))
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------- 5. Model ----------
class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Dense(sizes[i], sizes[i + 1], rng))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self):
        return [pg for layer in self.layers for pg in layer.params()]

    def predict(self, x):
        return self.forward(x).argmax(axis=1)

    def accuracy(self, x, y):
        return (self.predict(x) == y).mean()

    def fit(self, X, y, optimizer, epochs=50, batch_size=32,
            X_val=None, y_val=None, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        history = {"loss": [], "val_acc": []}
        for epoch in range(epochs):
            idx = rng.permutation(len(X))
            losses = []
            for i in range(0, len(X), batch_size):
                b = idx[i:i + batch_size]
                logits = self.forward(X[b])
                losses.append(self.loss_fn.forward(logits, y[b]))
                self.backward()
                optimizer.step(self.params())
            history["loss"].append(float(np.mean(losses)))
            if X_val is not None:
                history["val_acc"].append(float(self.accuracy(X_val, y_val)))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                va = f" val_acc={history['val_acc'][-1]:.3f}" if X_val is not None else ""
                print(f"epoch {epoch + 1:3d}  loss={history['loss'][-1]:.4f}{va}")
        return history