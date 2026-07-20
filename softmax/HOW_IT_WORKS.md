# Softmax Layer

The `Softmax` layer converts a vector of raw scores (called **logits**) into a probability distribution. The output values are always in the range **(0, 1)** and their sum is **1**.

---

## Formula

The softmax function is defined as

$$
p_i=\frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}
$$

where:

- $x_i$ is the input logit.
- $e^{x_i}$ converts every input value into a positive number.
- $\sum_{j=1}^{n}e^{x_j}$ computes the sum of all exponentiated values.
- $p_i$ is the probability assigned to class $i$.
- Dividing by this sum normalizes the outputs so they form a probability distribution.

The outputs satisfy

$$
\sum_{i=1}^{n}p_i=1
$$

---

## Numerical Stability

Directly computing

```python
np.exp(x)
```

can overflow for large values of `x`.

For example,

```python
np.exp(1000)
```

returns infinity.

To prevent this, the maximum value is subtracted before exponentiation.

```python
x_max = np.max(x, axis=self.axis, keepdims=True)
exp_x = np.exp(x - x_max)
```

This does **not** change the final probabilities because

$$
\frac{e^{x_i-c}}{\sum_j e^{x_j-c}}
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

for any constant $c$.

---

## Forward Pass

### Step 1

Convert the input into a NumPy array.

```python
x = np.asarray(x, dtype=np.float64)
```

---

### Step 2

Find the maximum value.

```python
x_max = np.max(x, axis=self.axis, keepdims=True)
```

---

### Step 3

Exponentiate the shifted logits.

```python
exp_x = np.exp(x - x_max)
```

---

### Step 4

Normalize them.

```python
self.output = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
```

The result is a probability distribution.

---

## Example

Input logits

```text
[2.0, 1.0, 0.1]
```

Subtract the maximum value

```text
[0.0, -1.0, -1.9]
```

Exponentiate

```text
[1.0000, 0.3679, 0.1496]
```

Normalize

```text
[0.6590, 0.2424, 0.0986]
```

Notice

```text
0.6590 + 0.2424 + 0.0986 = 1.0
```

---

## Backward Pass

The softmax layer has **no trainable parameters**, but it must still propagate gradients to the previous layer.

Suppose the network is

```text
Linear
   │
Logits
   │
Softmax
   │
Probabilities
   │
Loss
```

During backpropagation the loss provides

$$
\frac{\partial L}{\partial y}
$$

where $y$ is the softmax output.

The softmax layer must compute

$$
\frac{\partial L}{\partial x}
$$

where $x$ is the input logits.

---

## Softmax Jacobian

Unlike ReLU or Sigmoid, every softmax output depends on **every input**.

Its derivative is

$$
\frac{\partial y_i}{\partial x_j}=y_i(\delta_{ij}-y_j)
$$

where:

- $y_i$ is the softmax output.
- $\delta_{ij}$ is the Kronecker delta, which equals **1** if $i=j$ and **0** otherwise.

This produces an $N \times N$ Jacobian matrix.

Constructing this matrix explicitly is inefficient.

---

## Efficient Gradient Computation

Instead of forming the Jacobian, the gradient can be computed as

$$
\frac{\partial L}{\partial x}
=
y\odot\left(g-(g\cdot y)\right)
$$

where:

- $g=\frac{\partial L}{\partial y}$.
- $y$ is the softmax output.
- $\odot$ denotes element-wise multiplication.
- $g\cdot y$ denotes the dot product between the incoming gradient and the softmax output.

This is implemented as

```python
s = self.output

dot = np.sum(grad_output * s, axis=self.axis, keepdims=True)

grad_input = s * (grad_output - dot)
```

---

## Why is `backward()` Needed?

The softmax layer itself has **no trainable parameters**, so nothing inside it is updated.

However, the **previous layer** (for example, a Linear layer) needs gradients with respect to its outputs (the logits).

`Softmax.backward()` converts

$$
\frac{\partial L}{\partial \text{probabilities}}
$$

into

$$
\frac{\partial L}{\partial \text{logits}}
$$

so that earlier layers can compute gradients for their trainable parameters.

---

## Time Complexity

| Operation | Complexity |
|-----------|-----------:|
| Forward | $O(n)$ |
| Backward | $O(n)$ |

The implementation avoids constructing the full Jacobian, which would require $O(n^2)$ time and memory.

---

## Notes

- Supports vectors and batches of inputs.
- Uses numerically stable exponentiation.
- Stores the forward output for gradient computation.
- Computes the backward pass efficiently without explicitly building the Jacobian matrix.
- Commonly used as the final activation layer for multi-class classification problems.