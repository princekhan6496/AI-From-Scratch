# Cross Entropy Loss

## Overview

Cross Entropy Loss is one of the most commonly used loss functions for multi-class classification problems. It measures how well a model predicts the correct class by comparing the predicted class probabilities with the true class labels.

The input to Cross Entropy Loss is **logits**, which are the raw outputs of the model before any activation function is applied. Instead of explicitly computing the logarithm of the Softmax probabilities, the implementation computes **LogSoftmax** directly for improved numerical stability.

Mathematically, Cross Entropy Loss combines:

- **LogSoftmax**
- **Negative Log Likelihood (NLL)**

This is the same approach used internally by modern deep learning frameworks such as **PyTorch**.

---

## Computation Pipeline

```text
                 Logits
                    │
                    ▼
          Numerical Stabilization
                    │
                    ▼
             Stable Logits
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
     Softmax               LogSoftmax
        │                       │
        │                       ▼
        │            Select Correct Class
        │                       │
        └──────────────► Negative Log
                                │
                                ▼
                     Cross Entropy Loss
```

---

# Forward Pass

## Step 1: Receive Inputs

The loss receives:

- **Logits** of shape `(batch_size, num_classes)`
- **Targets** of shape `(batch_size,)`

Example:

```python
logits = np.array([
    [2.5, 1.2, 0.3],
    [0.4, 2.1, 1.0]
])

targets = np.array([0, 2])
```

Each row corresponds to one sample, and each column corresponds to one class.

---

## Step 2: Numerical Stabilization

Large logits can cause exponential overflow.

To prevent this, subtract the maximum value from every row.

```python
row_max = np.max(logits, axis=1, keepdims=True)

stable_logits = logits - row_max
```

Example:

Original logits:

```text
[1000, 1001, 1002]
```

Stable logits:

```text
[-2, -1, 0]
```

Subtracting the maximum value does **not** change the resulting Softmax probabilities.

---

## Step 3: Compute Softmax Probabilities

Compute exponentials of the stabilized logits.

```python
exp_logits = np.exp(stable_logits)
```

Compute the sum of exponentials.

```python
exp_sum = np.sum(exp_logits, axis=1, keepdims=True)
```

Normalize each row.

```python
probabilities = exp_logits / exp_sum
```

Mathematically,

$$
P_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

where

- \(P_i\) is the probability of class \(i\)
- \(z_i\) is the corresponding stabilized logit

Each row now represents a valid probability distribution whose values sum to **1**.

These probabilities are stored because they are required during backpropagation.

---

## Step 4: Compute LogSoftmax

Instead of computing

```python
np.log(probabilities)
```

the implementation computes LogSoftmax directly.

```python
log_softmax = stable_logits - np.log(exp_sum)
```

Mathematically,

$$
\log(\text{Softmax}(z)) = z-\log\left(\sum_j e^{z_j}\right)
$$

This formulation is mathematically equivalent to

```python
np.log(probabilities)
```

but is more numerically stable and avoids unnecessary computations.

---

## Step 5: Select the Correct Class

For every sample, only the log probability corresponding to the true class is required.

```python
sample_loss = -log_softmax[
    np.arange(targets.shape[0]),
    targets
]
```

Example:

LogSoftmax output

```text
[-0.18, -1.42, -2.71]
```

Target class

```text
0
```

Loss

```text
0.18
```

Only one value from each row contributes to the loss.

---

## Step 6: Apply Reduction

After computing the loss for every sample, the reduction method determines the final output.

### Mean

```python
loss = np.mean(sample_loss)
```

Returns the average loss across the batch.

---

### Sum

```python
loss = np.sum(sample_loss)
```

Returns the total loss.

---

### None

```python
loss = sample_loss
```

Returns the individual loss for every sample.

---

# Backward Pass

When Softmax and Cross Entropy are combined, the derivative simplifies significantly.

Instead of differentiating Softmax and Cross Entropy separately, the gradient becomes

$$
\frac{\partial L}{\partial z}=p-y
$$

where

- \(p\) is the predicted probability
- \(y\) is the one-hot encoded target

This simplification makes backpropagation both efficient and numerically stable.

---

## Step 1: Copy the Probabilities

```python
gradient = probabilities.copy()
```

---

## Step 2: Subtract One from the Correct Class

```python
gradient[
    np.arange(targets.shape[0]),
    targets
] -= 1
```

Example

Predicted probabilities

```text
[0.20, 0.70, 0.10]
```

Target

```text
Class 1
```

One-hot target

```text
[0, 1, 0]
```

Gradient

```text
[0.20, -0.30, 0.10]
```

Only the correct class is modified.

---

## Step 3: Mean Reduction

If the reduction mode is `"mean"`,

```python
gradient /= batch_size
```

This averages the gradients across the batch.

For `"sum"` and `"none"`, no additional scaling is performed.

---

# Algorithm

## Forward

```text
Receive logits and targets

        ↓

Subtract row-wise maximum

        ↓

Compute exponentials

        ↓

Compute Softmax probabilities

        ↓

Compute LogSoftmax

        ↓

Select target class log probability

        ↓

Take negative value

        ↓

Apply reduction

        ↓

Return loss
```

---

## Backward

```text
Copy Softmax probabilities

        ↓

Subtract 1 from target class

        ↓

If reduction == "mean"

        ↓

Divide by batch size

        ↓

Return gradient
```

---

# Time Complexity

## Forward Pass

```text
O(batch_size × num_classes)
```

---

## Backward Pass

```text
O(batch_size × num_classes)
```

---

## Memory Complexity

```text
O(batch_size × num_classes)
```

The implementation stores the Softmax probabilities for use during backpropagation.

---

# Numerical Stability

The implementation uses the **Log-Sum-Exp trick** to prevent numerical overflow.

Instead of computing

```python
np.exp(logits)
```

directly, it computes

```python
stable_logits = logits - np.max(logits, axis=1, keepdims=True)
```

This keeps exponentials within a safe numerical range while producing exactly the same Softmax probabilities.

This technique is used by modern deep learning frameworks such as PyTorch.

---

# Summary

This implementation performs the following steps:

1. Receive logits and target labels.
2. Stabilize the logits by subtracting the maximum value in each row.
3. Compute Softmax probabilities.
4. Compute LogSoftmax directly for numerical stability.
5. Select the log probability corresponding to the correct class.
6. Compute the Negative Log Likelihood (NLL).
7. Apply the requested reduction (`mean`, `sum`, or `none`).
8. During backpropagation, compute the gradient using

   $$
   \text{gradient} = \text{probabilities} - \text{one\_hot(target)}
   $$

9. Divide by the batch size when using mean reduction.

This implementation is fully vectorized, numerically stable, efficient, and mathematically equivalent to the standard Cross Entropy Loss used in modern deep learning frameworks.