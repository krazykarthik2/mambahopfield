# Final Findings: Memory Retrieval Complexity and Accuracy Trade-offs

This document provides a comprehensive comparison of different memory retrieval models evaluated during our investigation, focusing on their time complexity, space complexity, and retrieval accuracy (MSE).

## Summary Table

| Model Configuration | Retrieval Time Complexity (per token) | Space Complexity (Weights/Memory Storage) | Exact Retrieval? (MSE = 0.0) | Resolution on Erased/Noisy Queries |
| :--- | :---: | :---: | :---: | :--- |
| **Standard Self-Attention** | $O(N \cdot d)$ | $O(N \cdot d)$ (KV Cache scales with sequence $N$) | **Yes** | Perfect reconstruction. High sequence length cost. |
| **Dual-Projection Hopfield (Mamba-style Scan)** | $O(d \cdot M)$ | $O(d \cdot M)$ (Fixed size, independent of $N$) | **Yes** | Perfect reconstruction. Scales linearly with number of memories $M$. |
| **Pure Linear Pseudoinverse Hopfield** | $O(d^2)$ | $O(d^2)$ (Fixed size matrix $\mathbf{S}$) | **No** (Yes only for noisy) | Perfect on noisy queries, but experiences class confusion on erased queries (MSE $\approx 0.10$). |
| **Centered Polynomial Hopfield** | $O(d^2)$ | $O(d^2)$ (Fixed size matrix $\mathbf{S}$) | **No** | Blurred reconstruction (MSE $\approx 0.08 - 0.13$) due to coordinate distortions. |
| **Standard Linear Hopfield (ELU)** | $O(d^2)$ | $O(d^2)$ (Fixed size matrix $\mathbf{S}$) | **No** | Blurred average output due to crosstalk/overlap (MSE $\approx 0.09$). |

---

## Key Takeaways

### 1. The Attention Bottleneck
Standard Self-Attention achieves perfect retrieval accuracy by keeping the entire history in the KV Cache. However, this causes the computation cost per token to grow linearly with sequence length $N$, leading to quadratic complexity $O(N^2 \cdot d)$ over a sequence.

### 2. The Selective Scan Compromise
By implementing the **Dual-Projection Hopfield** update within a simulated Mamba selective scan block, we achieve **exact MSE = 0.0 retrieval** with sequence time complexity that scales linearly $O(N \cdot d \cdot M)$. This replaces the sequence dependency $N$ with the memory size dependency $M$.

### 3. The Linear representation Limit ($O(d^2)$)
Compressing memories into a single fixed-size projection matrix $\mathbf{S} \in \mathbb{R}^{d \times d}$ removes dependency on both sequence length $N$ and the number of memories $M$. While this is highly efficient, the lack of non-linear selection (softmax) means that erased query information loss cannot be recovered perfectly (stalling at MSE $\approx 0.10$).
