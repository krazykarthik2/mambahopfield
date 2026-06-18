# Architectural Comparison: Modern Hopfield Networks vs. Static Cross-Attention

This document provides a direct comparison between the **original continuous Modern Hopfield Network (MHN)** and our **Static Cross-Attention Hopfield** architecture.

---

## 1. Parameters and Complexity Table

Let:
* $N$ = Input sequence length
* $d$ = Feature dimension (parameter width per token)
* $M$ = Number of stored memories (static templates)

| Metric | Original Modern Hopfield (MHN) | Static Cross-Attention Hopfield |
| :--- | :---: | :---: |
| **Total Parameter Count** | **$d \cdot N$** (Stored dynamically in KV activation cache) | **$d \cdot M$** (Stored statically in Key-Value weights $\mathbf{K}, \mathbf{V}$) |
| **Time Complexity (per token)** | $O(N \cdot d)$ (Grows as sequence progresses) | **$O(d \cdot M)$** (Constant runtime per token) |
| **Sequence Time Complexity** | **Quadratic $O(N^2 \cdot d)$** | **Linear $O(N \cdot d \cdot M)$** |
| **Activation Memory Space** | $O(N \cdot d)$ (KV cache must scale with sequence $N$) | **$O(d \cdot M)$** (Static; independent of sequence length $N$) |
| **Softmax Scaling Dimension** | Sequence length $N$ | Memory bank size $M$ |
| **Exponential Capacity ($2^{d/2}$)** | **Yes** | **Yes** |
| **Exact Retrieval (MSE = 0.0)** | **Yes** | **Yes** |

---

## 2. Key Architectural Differences

### 1. Parameter Storage and Scaling
* **Original MHN**: Stored patterns are held dynamically in the KV cache, scaling with sequence length ($d \cdot N$). The parameter footprint grows continuously as the sequence length increases.
* **Static Cross-Attention**: Stored patterns are held in fixed weight matrices ($\mathbf{K} \in \mathbb{R}^{M \times d}$ and $\mathbf{V} \in \mathbb{R}^{d \times M}$). The parameter footprint is static ($d \cdot M$) and independent of sequence length.

### 2. Computational Bottleneck
* **Original MHN**: The softmax normalized attention operates over the sequence dimension $N$. This introduces the classic quadratic attention bottleneck $O(N^2)$, making long-sequence inference expensive.
* **Static Cross-Attention**: The softmax normalized attention operates over the static memory count $M$. Because $M$ is fixed, processing scales strictly linearly with the sequence length $O(N)$, allowing fast inference over arbitrary sequence lengths.

