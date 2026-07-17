import math

class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float))

        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = math.exp(self.data)

        out = Value(x, (self,), "exp")

        def _backward():
            self.grad += x * out.grad

        out._backward = _backward
        return out

    def log(self):
        if self.data <= 0:
            raise ValueError("log undefined for non-positive values")

        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)

        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))

        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def abs(self):
        out = Value(abs(self.data), (self,), "abs")

        def _backward():
            if self.data > 0:
                self.grad += out.grad
            elif self.data < 0:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def sqrt(self):
        out = Value(math.sqrt(self.data), (self,), "sqrt")

        def _backward():
            self.grad += (0.5 / math.sqrt(self.data)) * out.grad

        out._backward = _backward
        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")

        def _backward():
            self.grad += math.cos(self.data) * out.grad

        out._backward = _backward
        return out

    def cos(self):
        out = Value(math.cos(self.data), (self,), "cos")

        def _backward():
            self.grad += -math.sin(self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        for node in topo:
            node.grad = 0.0

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        for node in topo:
            node.grad = 0.0

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Value(other) * (self ** -1)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"