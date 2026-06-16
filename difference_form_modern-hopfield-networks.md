# Differences: Mamba Dual-Projection vs. Original Modern Hopfield Networks

This document details the mathematical and computational differences between our **Mamba-style Dual-Projection Hopfield Network** and the **original continuous Modern Hopfield Network (MHN)**.

---

## 1. Formulation Comparison

### Original Modern Hopfield Network (MHN)
The original continuous MHN (Ramsauer et al.) retrieves memory states using the update rule:
$$\mathbf{\xi}^{(t+1)} = \mathbf{X} \text{Softmax}\left(\beta \mathbf{X}^T \mathbf{\xi}^{(t)}\right)$$

When integrated into transformer architectures, the memory bank $\mathbf{X}$ is formed by the history of the sequence itself. Therefore, the number of patterns $M$ scales directly with the sequence length $N$ ($M = N$).

### Mamba-style Dual-Projection Hopfield
Our formulation decouples the active sequence length $N$ from the stored memories $\mathbf{X} \in \mathbb{R}^{d \times M}$:
1. **Projection**: $\mathbf{z} = \mathbf{X}^T \mathbf{\xi}^{(t)} \in \mathbb{R}^M$
2. **Routing (Softmax)**: $\mathbf{w} = \text{Softmax}(\beta \mathbf{z}) \in \mathbb{R}^M$
3. **Reconstruction**: $\mathbf{\xi}^{(t+1)} = \mathbf{X} \mathbf{w} \in \mathbb{R}^d$

Here, $\mathbf{X}$ is a fixed-size memory bank (e.g. $M = 10$ static class templates) rather than the dynamic sequence history.

---

## 2. Complexity and Scaling Comparison

| Feature | Original Modern Hopfield (MHN) | Mamba Dual-Projection Hopfield |
| :--- | :---: | :---: |
| **Sequence Time Complexity** | **Quadratic $O(N^2 \cdot d)$** | **Linear $O(N \cdot d \cdot M)$** |
| **Retrieval Complexity (per token)** | $O(N \cdot d)$ (grows with sequence) | $O(d \cdot M)$ (constant per token) |
| **Memory scaling $M$** | $M = N$ (tied to sequence length) | $M \ll N$ (decoupled static memory bank) |
| **Softmax Dimension** | Sequence history $N$ | Stored memories $M$ |
| **Exact Retrieval? (MSE = 0.0)** | **Yes** | **Yes** |

---

## 3. Key Architectural Advantages

1. **Linear Sequence Scaling**: By routing tokens to a static associative memory bank instead of attending to all past tokens, our model bypasses the quadratic bottleneck $O(N^2)$ of Transformers, making it compatible with Mamba's linear-time sequence processing.
2. **Constant Compute Footprint**: The computational requirement per token is $O(d \cdot M)$. Since $M$ is fixed, processing the 1000th token in a sequence takes the exact same time and memory as processing the 1st token.
