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
| **Model Parameters** | $0$ (No weights; it computes self-similarities directly) | **$d \cdot M$** (Key-Value weights: $\mathbf{K} \in \mathbb{R}^{M \times d}$, $\mathbf{V} \in \mathbb{R}^{d \times M}$) |
| **Time Complexity (per token)** | $O(N \cdot d)$ (Grows as sequence progresses) | **$O(d \cdot M)$** (Constant runtime per token) |
| **Sequence Time Complexity** | **Quadratic $O(N^2 \cdot d)$** | **Linear $O(N \cdot d \cdot M)$** |
| **Activation Memory Space** | $O(N \cdot d)$ (KV cache must scale with sequence $N$) | **$O(d \cdot M)$** (Static; independent of sequence length $N$) |
| **Softmax Scaling Dimension** | Sequence length $N$ | Memory bank size $M$ |
| **Exponential Capacity ($2^{d/2}$)** | **Yes** | **Yes** |
| **Exact Retrieval (MSE = 0.0)** | **Yes** | **Yes** |

---

## 2. Key Architectural Differences

### 1. Parametric vs. Non-Parametric Memory
* **Original MHN**: Non-parametric. The associative memory bank $\mathbf{X}$ is populated dynamically by the sequence tokens themselves ($M = N$). There are no learnable weights for the memories.
* **Static Cross-Attention**: Parametric. Stored patterns are held in fixed weight matrices ($\mathbf{K} \in \mathbb{R}^{M \times d}$ and $\mathbf{V} \in \mathbb{R}^{d \times M}$). The parameters are learnable or pre-loaded, rather than collected from sequence history.

### 2. Computational Bottleneck
* **Original MHN**: The softmax normalized attention operates over the sequence dimension $N$. This introduces the classic quadratic attention bottleneck $O(N^2)$, making long-sequence inference expensive.
* **Static Cross-Attention**: The softmax normalized attention operates over the static memory count $M$. Because $M$ is fixed, processing scales strictly linearly with the sequence length $O(N)$, allowing fast inference over arbitrary sequence lengths.
