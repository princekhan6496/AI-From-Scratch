# Embedding Layer From Scratch (NumPy)

A complete implementation of a trainable **Embedding Layer** from scratch using **NumPy**, inspired by the mathematical foundations behind `torch.nn.Embedding`.

This project is designed to understand **how embedding layers actually work internally**, instead of simply using a deep learning framework.

No PyTorch is used.

---

## Motivation

Most deep learning libraries provide an embedding layer as a single function call:

```python
embedding = nn.Embedding(vocab_size, embedding_dim)
```

While easy to use, this hides what is actually happening.

This project rebuilds the embedding layer from first principles to understand:

- Why embeddings exist
- Why one-hot vectors are inefficient
- How embedding lookup works
- Why `weight[token_id]` is mathematically equivalent to one-hot multiplication
- How modern frameworks optimize embedding lookup

---

# Mathematical Background

An embedding layer is mathematically defined as

```text
One-Hot Vector × Embedding Matrix = Embedding Vector
```

Example

Vocabulary Size = 4

```
Token "b"

↓

One-Hot

[0 0 1 0]
```

Embedding Matrix

```
[
 [0.2 -0.5]
 [1.0  0.4]
 [-0.7 2.1]
 [0.5 -1.2]
]
```

Result

```
[0 0 1 0] × Embedding Matrix = [-0.7 2.1]
```

Since a one-hot vector contains only a single `1`, this multiplication simply selects one row of the embedding matrix.

Therefore,

```text
One-Hot × Weight Matrix
```

is mathematically identical to

```python
weight[token_id]
```

Modern deep learning frameworks use this optimized lookup instead of explicitly generating one-hot vectors.

---

# Features

- Built entirely with NumPy
- No PyTorch
- Random weight initialization
- Supports arbitrary vocabulary size
- Supports arbitrary embedding dimensions
- Supports batches
- Supports sequences
- Performs input validation
- PyTorch-like API

---

# Project Structure

```
Embedding/
│
├── embedding.py
├── example.py
├── README.md
```

---

# API

## Create an Embedding Layer

```python
embedding = Embedding(
    vocab_size=50000,
    embedding_dim=768
)
```

---

## Input

Input is a matrix of token IDs.

```python
tokens = np.array([
    [12, 45, 91],
    [ 5,  8, 20]
])
```

Shape

```
(batch_size, sequence_length)
```

Example

```
(2,3)
```

---

## Output

```python
embeddings = embedding(tokens)
```

Output Shape

```
(batch_size, sequence_length, embedding_dim)
```

Example

```
(2,3,768)
```

Each token ID is replaced with its corresponding embedding vector.

---

# Internal Weight Matrix

The embedding layer stores exactly one trainable parameter.

```
Weight Matrix

Shape

(vocab_size, embedding_dim)
```

Example

```
(50000,768)
```

Each row represents one token.

```
Row 0  → Token 0

Row 1  → Token 1

Row 2  → Token 2

...

Row 49999 → Token 49999
```

---

# Forward Pass

The forward pass is simply

```python
return self.weight[token_ids]
```

Conceptually NumPy performs

```python
output = []

for sentence in token_ids:

    row = []

    for token in sentence:

        row.append(weight[token])

    output.append(row)
```

The actual implementation uses NumPy's optimized advanced indexing.

---

# Why Not One-Hot Encoding?

The naive implementation performs

```text
Token ID

↓

One-Hot

↓

Matrix Multiplication

↓

Embedding
```

Example

Vocabulary = 50000

```
One-Hot Vector

[0 0 0 ... 1 ... 0]
```

This vector contains

```
49999 zeros

1 one
```

Constructing and multiplying such vectors is computationally wasteful.

Instead, frameworks directly retrieve the required row from the embedding matrix.

```
Token ID

↓

Embedding Matrix

↓

Row Lookup

↓

Embedding Vector
```

This produces exactly the same result while being significantly faster and more memory efficient.

---

# Complexity

Naive One-Hot Method

```
Time

O(V × D)
```

Memory

```
O(V)
```

Where

- V = Vocabulary Size
- D = Embedding Dimension

---

Optimized Lookup

Time

```
O(D)
```

Memory

```
O(1)
```

This optimization is why every modern deep learning framework implements embeddings using row lookup.

---

# Comparison with PyTorch

| Feature | This Project | PyTorch |
|----------|-------------|----------|
| NumPy Only | ✅ | ❌ |
| Forward Pass | ✅ | ✅ |
| Batch Support | ✅ | ✅ |
| Sequence Support | ✅ | ✅ |
| Input Validation | ✅ | ✅ |
| Trainable Weights | ✅ | ✅ |
| Automatic Gradients | ❌ | ✅ |
| Sparse Gradients | ❌ | ✅ |

---

# Example

```python
import numpy as np
from embedding import Embedding

embedding = Embedding(
    vocab_size=10000,
    embedding_dim=128
)

tokens = np.array([
    [1,5,10],
    [2,8,3]
])

output = embedding(tokens)

print(output.shape)
```

Output

```
(2,3,128)
```