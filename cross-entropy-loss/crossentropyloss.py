import numpy as np


class CrossEntropyLoss:

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def forward(self, logits, targets):

        self.logits = logits
        self.targets = targets

        shifted = logits - np.max(logits, axis=1, keepdims=True)

        exp = np.exp(shifted)

        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        log_probs = shifted - np.log(np.sum(exp, axis=1, keepdims=True))

        loss = -log_probs[np.arange(len(targets)), targets]

        if self.reduction == "mean":
            return np.mean(loss)

        if self.reduction == "sum":
            return np.sum(loss)

        return loss

    def backward(self):

        grad = self.probs.copy()

        grad[np.arange(len(self.targets)), self.targets] -= 1

        if self.reduction == "mean":
            grad /= len(self.targets)

        return grad