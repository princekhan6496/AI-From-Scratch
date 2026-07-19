import numpy as np


class Embedding:

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        init_std=0.02
    ):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.weight = np.random.randn(
            vocab_size,
            embedding_dim
        ) * init_std

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        x = np.asarray(x, dtype=np.int64)

        if np.any(x >= self.vocab_size):
            raise ValueError("Token id exceeds vocabulary size")

        if np.any(x < 0):
            raise ValueError("Negative token ids are not allowed")

        return self.weight[x]