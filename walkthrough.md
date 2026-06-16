# Walkthrough of Mamba Exact Retrieval

We evaluated the low-rank dual-projection modern Hopfield formulation inside a Mamba-style selective sequence scan block.

## Implementation Details
- **Script**: [mamba_exact_hopfield.py](file:///C:/Users/karthikkrazy/Documents/antigravity/optimistic-lovelace/mamba_exact_hopfield.py)
- **Methodology**: Maps sequence states sequentially:
  1. Projection: $\mathbf{z} = \mathbf{X}^T \mathbf{x}$
  2. Routing: $\mathbf{w} = \text{Softmax}(\beta \mathbf{z})$
  3. Reconstruction: $\mathbf{x}' = \mathbf{X} \mathbf{w}$
  
- **Visual Verification**: Output stored in `mamba_exact_results.png`.

## Final Results Table

| Query Configuration | Final MSE | Exact Pixel Retrieval? | Complexity |
| :--- | :---: | :---: | :---: |
| **Erased Query** (Bottom 50% Zeroed) | **0.00000000** | **Yes** | $O(d \cdot M)$ |
| **Noisy Query** (Gaussian Noise $\sigma=0.6$) | **0.00000000** | **Yes** | $O(d \cdot M)$ |
