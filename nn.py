import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [(rng.normal(0, np.sqrt(2 / a), (a, b))).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self.steps = 0

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts

    def predict(self, X):
        return self.forward(X)[-1]

    def train_step(self, X, Y, M, w, lr=1e-3):
        acts = self.forward(X)
        diff = (acts[-1] - Y) * M
        denom = max(float(M.sum()), 1.0)
        loss = float((w[:, None] * diff ** 2).sum() / denom)
        g = 2 * w[:, None] * diff / denom
        grads = [None] * (2 * len(self.W))
        for i in reversed(range(len(self.W))):
            grads[i] = acts[i].T @ g
            grads[len(self.W) + i] = g.sum(0)
            if i:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        self.steps += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, (p, gr) in enumerate(zip(self.W + self.b, grads)):
            self._m[k] = b1 * self._m[k] + (1 - b1) * gr
            self._v[k] = b2 * self._v[k] + (1 - b2) * gr * gr
            mh = self._m[k] / (1 - b1 ** self.steps)
            vh = self._v[k] / (1 - b2 ** self.steps)
            p -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        return loss

    def arrays(self, prefix):
        d = {f"{prefix}W{i}": W for i, W in enumerate(self.W)}
        d.update({f"{prefix}b{i}": b for i, b in enumerate(self.b)})
        return d

    @classmethod
    def from_arrays(cls, data, prefix):
        n = sum(1 for k in data if k.startswith(prefix + "W"))
        W = [np.array(data[f"{prefix}W{i}"]) for i in range(n)]
        net = cls([W[0].shape[0]] + [w.shape[1] for w in W])
        net.W, net.b = W, [np.array(data[f"{prefix}b{i}"]) for i in range(n)]
        return net


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    net = MLP([1, 32, 32, 2])
    X = rng.uniform(-1, 1, (512, 1)).astype(np.float32)
    Y = np.hstack([np.sin(3 * X), np.zeros_like(X)]).astype(np.float32)
    M = np.array([[1, 0]] * 512, np.float32)
    w = np.ones(512, np.float32)
    first = net.train_step(X, Y, M, w)
    for _ in range(1500):
        last = net.train_step(X, Y, M, w, lr=3e-3)
    print(f"loss {first:.4f} -> {last:.5f}")
    assert last < 0.01 * first
    print("nn self-check passed")
