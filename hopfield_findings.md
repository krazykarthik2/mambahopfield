# Findings: Linear Hopfield (Mamba-Style) Memory Bank

We successfully implemented a **Linear Hopfield Memory Bank** using a state-space/linear-attention formulation, compressing the key-value prototype lookups into a single fixed $32 \times 32$ matrix $\mathbf{S}$. We compared it against standard quadratic softmax-based Hopfield retrieval.

---

## Performance Summary (after 10 epochs)

| Configuration | Validation Loss | Validation Accuracy (%) | Retrieval Complexity |
| :--- | :---: | :---: | :---: |
| **Softmax Hopfield (Centered)** | **1.0506** | **64.00%** | $O(M)$ time/space |
| **Baseline (No Memory)** | 1.1072 | 62.25% | - |
| **Linear Hopfield (Mamba-Style)** | 1.2056 | 61.20% | **$O(1)$ time/space** |

### Performance Visualization

![Validation Curves for Hopfield Experiment](C:/Users/karthikkrazy/.gemini/antigravity/brain/0241d110-ba49-44a7-8ea2-0adac73b6146/hopfield_results.png)

---

## Detailed Analysis: The Complexity vs. Capacity Tradeoff

### 1. Compression and Retrieval Speed ($O(1)$ Complexity)
In the **Linear Hopfield** model, the key-value memory lookup is mathematically reformulated:
- Instead of keeping the memory matrix $\mathbf{M}$ as a list of vectors and comparing them at runtime ($O(M)$ complexity where $M=10$), we pre-computed the memory state matrix $\mathbf{S} \in \mathbb{R}^{32 \times 32}$.
- At runtime, query retrieval is a single vector-matrix multiplication: $\mathbf{z}_{retrieved} = \phi(\mathbf{q}) \mathbf{S}$, executing in **$O(1)$ constant time** relative to memory size.
- This is the exact recurrent state-space mechanism that allows Mamba/SSMs to achieve linear $O(N)$ sequence length scaling.

### 2. Why the Linear Hopfield Underperformed (The Uncentered Shift)
The `softmax_hopfield` configuration (64.00%) explicitly subtracted the mean latent vector to resolve the **ReLU Alignment Problem** before computing cosine similarity.
In the `linear_hopfield` configuration:
- We applied the feature map $\phi(\mathbf{x}) = \text{elu}(\mathbf{x}) + 1.0$ directly on the raw non-negative encoder outputs.
- Because $\mathbf{z} \geq 0$, the feature map shifted all values upward ($\geq 1.0$). 
- This reintroduced the positive alignment problem at the linear attention bottleneck. The denominator $\phi(\mathbf{q})\mathbf{z}_{norm}^T$ ended up scaling the retrieved vector uniformly, causing the retrieved representation to collapse into a near-constant vector that acted as a noisy distractor, reducing final classification performance to **61.20%**.

---

## The Path Forward: Centering the State Space
To make the Mamba-style Linear Hopfield bank work as effectively as Softmax Hopfield, the latent representations must be centered **before** compiling the memory matrix $\mathbf{S}$:
1. Center keys and queries: $\mathbf{k}^{centered}_i = \mathbf{k}_i - \mathbf{\mu}$
2. Compile centered memory matrix: $\mathbf{S}^{centered} = \sum \phi(\mathbf{k}^{centered}_i)^T \mathbf{v}^{centered}_i$
3. Retrieve using centered query: $\mathbf{z}_{ret = } = \phi(\mathbf{q} - \mathbf{\mu}) \mathbf{S}^{centered} / \text{norm}$
