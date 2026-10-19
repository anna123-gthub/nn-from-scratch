"""Train on sklearn's 8x8 digits (offline). Compare optimizers and plot curves."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from nn import MLP, SGD, Momentum, Adam

X, y = load_digits(return_X_y=True)
X = X / 16.0  # scale pixels to [0, 1]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

results = {}
for name, opt in [("SGD", SGD(0.1)), ("Momentum", Momentum(0.05)), ("Adam", Adam(0.003))]:
    print(f"\n=== {name} ===")
    net = MLP([64, 64, 10], seed=0)
    h = net.fit(X_tr, y_tr, opt, epochs=30, batch_size=32, X_val=X_te, y_val=y_te)
    results[name] = (h, net.accuracy(X_te, y_te))

print("\nFinal test accuracy:")
for name, (_, acc) in results.items():
    print(f"  {name:9s} {acc:.3f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for name, (h, _) in results.items():
        plt.plot(h["loss"], label=name)
    plt.xlabel("epoch"); plt.ylabel("train loss"); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    print("saved loss_curves.png")
except ImportError:
    pass