"""Linear probe: logistic regression on one layer's hidden state, written
from scratch in PyTorch.

  p(wrong | h) = sigmoid(w . standardize(h) + b)

Why linear on purpose: if a single linear direction in activation space
separates "about to be right" from "about to be wrong", that is evidence
the model represents its own uncertainty explicitly. An MLP probe (Phase 3)
could learn the answer on its own and prove much less.

Leakage rules:
  - mean/std for standardization are computed on the TRAIN split only.
  - the L2 strength is chosen on a validation slice of train, never test.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-6


class LinearProbe:
    def __init__(self, l2: float = 1e-2, max_iter: int = 200, seed: int = 0):
        self.l2 = l2
        self.max_iter = max_iter
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.w_: torch.Tensor | None = None
        self.b_: torch.Tensor | None = None

    def _standardize(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((X - self.mean_) / self.std_, dtype=torch.float32)

    def fit(self, X: np.ndarray, y_wrong: np.ndarray) -> LinearProbe:
        """X: (n, d) activations from one layer. y_wrong: (n,) 0/1."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_wrong)
        if len(np.unique(y)) < 2:
            raise ValueError("probe needs both right and wrong examples to train")

        torch.manual_seed(self.seed)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + EPS
        Xt = self._standardize(X)
        yt = torch.as_tensor(y, dtype=torch.float32)

        d = Xt.shape[1]
        w = torch.zeros(d, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        # LBFGS: full-batch, deterministic, no learning-rate tuning --
        # the standard way to fit a small convex model like this.
        opt = torch.optim.LBFGS([w, b], max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logits = Xt @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
            loss = loss + self.l2 * (w**2).sum()  # bias not regularized
            loss.backward()
            return loss

        opt.step(closure)
        self.w_, self.b_ = w.detach(), b.detach()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """p(wrong) for each row -- this IS the risk score."""
        if self.w_ is None:
            raise RuntimeError("call fit() before predict_proba()")
        logits = self._standardize(np.asarray(X, dtype=np.float64)) @ self.w_ + self.b_
        return torch.sigmoid(logits).numpy()

    def direction(self) -> np.ndarray:
        """The learned 'I'm about to be wrong' direction in (standardized)
        activation space. Useful later for cross-model / cross-domain
        analysis."""
        if self.w_ is None:
            raise RuntimeError("call fit() first")
        return self.w_.numpy().copy()


def select_l2(X: np.ndarray, y: np.ndarray, grid=(1e-3, 1e-2, 1e-1, 1.0), val_frac=0.25, seed=0):
    """Pick L2 strength on a validation slice OF TRAIN by AUROC. Test data
    never touches this function."""
    from halluscope.eval.metrics import auroc

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = max(2, int(len(y) * val_frac))
    val, tr = idx[:n_val], idx[n_val:]
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[val])) < 2:
        return grid[len(grid) // 2]  # not enough of both classes to choose; fall back to middle

    best_l2, best_auc = grid[0], -1.0
    for l2 in grid:
        probe = LinearProbe(l2=l2, seed=seed).fit(X[tr], y[tr])
        auc = auroc(y[val], probe.predict_proba(X[val]))
        if auc > best_auc:
            best_l2, best_auc = l2, auc
    return best_l2
