import numpy as np

class Softmax:
    def __init__(self, axis=-1):
        self.axis = axis
        self.output = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)

        x_max = np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.output = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        return self.output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)

        s = self.output

        dot = np.sum(grad_output * s, axis=self.axis, keepdims=True)

        grad_input = s * (grad_output - dot)

        return grad_input