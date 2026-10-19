"""Verify backprop against numerical gradients."""
import numpy as np
from nn import MLP

rng = np.random.default_rng(1)
X = rng.standard_normal((5, 4))
y = rng.integers(0, 3, size=5)
net = MLP([4, 6, 3], seed=2)

# analytic gradients (from your backprop)
net.loss_fn.forward(net.forward(X), y)
net.backward()


def loss():
    return net.loss_fn.forward(net.forward(X), y)


# numerical gradients (nudge each weight a tiny bit)
eps, worst = 1e-5, 0.0
for p, g in net.params():
    for idx in np.ndindex(p.shape):
        old = p[idx]
        p[idx] = old + eps
        lp = loss()
        p[idx] = old - eps
        lm = loss()
        p[idx] = old
        num = (lp - lm) / (2 * eps)
        rel = abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx]))
        worst = max(worst, rel)

print(f"max relative error: {worst:.2e}")
assert worst < 1e-4, "backprop has a bug"
print("gradient check PASSED")