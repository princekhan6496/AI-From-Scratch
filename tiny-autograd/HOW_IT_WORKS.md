# How tiny-autograd Works

## About

**tiny-autograd** is my from-scratch implementation of a tiny reverse-mode automatic differentiation engine built to understand the mathematical foundations of neural networks and backpropagation.

Instead of relying on existing deep learning frameworks, this project implements automatic differentiation from first principles using a computational graph, reverse-mode differentiation, and the Chain Rule.

The engine supports scalar automatic differentiation and demonstrates how gradients are computed through arbitrary mathematical expressions. Although intentionally lightweight, it follows the same core principles used in modern frameworks such as **PyTorch**, **TensorFlow**, and **JAX**.

---

# Features

- Reverse-mode automatic differentiation
- Dynamic computational graph
- Automatic backpropagation
- Topological graph traversal
- Gradient accumulation
- Scalar computation engine
- Operator overloading
- Mathematical operations:
  - Addition
  - Subtraction
  - Multiplication
  - Division
  - Power
  - Negation
- Activation functions:
  - ReLU
  - Sigmoid
  - Tanh
- Mathematical functions:
  - Exponential
  - Logarithm
  - Square Root
  - Absolute Value
  - Sine
  - Cosine
- Automatic gradient reset using `zero_grad()`
- Pure Python implementation

---

# Example

```python
a = Value(2)
b = Value(-3)
c = Value(10)

d = a * b
e = d + c
f = e.relu()
g = f ** 2
h = g + a

h.backward()
```

---

# Computational Graph

The above expression creates the following computational graph.

![Computational Graph](Images\d1.png)

Every node in the graph is represented by a `Value` object.

Each node stores:

- `data` — numerical value
- `grad` — accumulated gradient
- `_prev` — parent nodes
- `_op` — operation that produced the node
- `_backward` — local gradient function

---

# Forward Pass

The forward pass evaluates the expression from the inputs to the final output.

| Variable | Value |
|----------|------:|
| a | 2 |
| b | -3 |
| c | 10 |
| d = a × b | -6 |
| e = d + c | 4 |
| f = ReLU(e) | 4 |
| g = f² | 16 |
| h = g + a | 18 |

Final output

```text
18
```

---

# Building the Computational Graph

Each mathematical operation creates a new `Value`.

For example,

```python
d = a * b
```

creates

```
a ----\
       *
b ----/
```

The multiplication node stores

- its output value
- references to `a` and `b`
- its local backward function

No gradients are computed during the forward pass.

The graph simply records how every value was produced.

---

# Local Derivatives

Every operation knows only its own derivative.

## Addition

```text
z = x + y
```

```
∂z/∂x = 1

∂z/∂y = 1
```

---

## Multiplication

```text
z = x × y
```

```
∂z/∂x = y

∂z/∂y = x
```

---

## Division

```text
z = x / y
```

Internally,

```text
x × y⁻¹
```

---

## Power

```text
z = xⁿ
```

```
∂z/∂x = n·xⁿ⁻¹
```

---

## ReLU

```text
ReLU(x)
```

```
x > 0  → 1

x ≤ 0  → 0
```

---

## Sigmoid

```text
σ(x) = 1 / (1 + e⁻ˣ)
```

```
σ'(x) = σ(x)(1-σ(x))
```

---

## Tanh

```text
tanh(x)
```

```
1 - tanh²(x)
```

---

## Exponential

```text
exp(x)
```

```
d/dx exp(x) = exp(x)
```

---

## Logarithm

```text
log(x)
```

```
d/dx log(x) = 1/x
```

---

## Square Root

```text
√x
```

```
d/dx √x = 1/(2√x)
```

---

## Sine

```text
sin(x)
```

```
d/dx sin(x) = cos(x)
```

---

## Cosine

```text
cos(x)
```

```
d/dx cos(x) = -sin(x)
```

---

## Absolute Value

```text
|x|
```

```
x > 0 → 1

x < 0 → -1
```

---

# Backpropagation

Backpropagation begins from the final output.

```python
h.backward()
```

The output gradient is initialized as

```text
h.grad = 1
```

because

```text
∂h/∂h = 1
```

Each node then executes its stored `_backward()` function to distribute gradients to its parent nodes.

---

# Chain Rule

tiny-autograd computes derivatives using the Chain Rule.

Instead of computing one enormous derivative directly, every node computes

```
incoming gradient
        ×
local derivative
```

and passes the result to its parents.

This process continues until every leaf node receives its gradient.

---

# Topological Ordering

Gradients cannot be computed in arbitrary order.

tiny-autograd first performs a depth-first traversal of the graph to build a topological ordering.

Forward construction

```text
a
b
d
c
e
f
g
h
```

Reverse traversal used during backpropagation

```text
h
g
f
e
d
c
b
a
```

This guarantees that every node has already received gradients from all of its children before propagating gradients further backward.

---

# Gradient Accumulation

Variables may contribute to the output through multiple paths.

For example,

```python
a * b
```

and

```python
g + a
```

both depend on `a`.

Therefore gradients are accumulated using

```python
self.grad += ...
```

instead of

```python
self.grad = ...
```

This correctly sums gradient contributions from every computational path.

---

# Resetting Gradients

Gradients accumulate across multiple backward passes.

To clear previously computed gradients,

```python
h.zero_grad()
```

or simply call

```python
h.backward()
```

if your implementation automatically resets gradients before backpropagation.

---

# Final Gradients

After

```python
h.backward()
```

the gradients become

| Variable | Gradient |
|----------|---------:|
| a | -23 |
| b | 16 |
| c | 8 |

These gradients tell us how much the final output changes with respect to each input. For example, a small increase in a changes the output by approximately -23 × Δa, while small increases in b and c change the output by approximately 16 × Δb and 8 × Δc, respectively.

---

# Project Structure

```
Value
│
├── data
├── grad
├── _prev
├── _op
└── _backward()
```

Every operation creates another `Value`, gradually building a directed acyclic graph (DAG).

---

# Conclusion

tiny-autograd is built around five fundamental ideas:

1. Compute the forward value.
2. Record the parent nodes.
3. Store the local derivative.
4. Traverse the graph in reverse topological order.
5. Apply the Chain Rule to compute gradients.

Although only a lightweight scalar autograd engine, tiny-autograd demonstrates the exact principles that power modern automatic differentiation systems used in deep learning libraries such as **PyTorch**, **TensorFlow**, and **JAX**. It serves as a foundation for understanding how neural networks learn through backpropagation.

