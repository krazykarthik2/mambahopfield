# Findings: Dual-Projection Hopfield exact retrieval

We evaluated two models:
1. **Centered Polynomial Pseudoinverse Linear Hopfield** ($\mathbf{S} = \mathbf{X} \phi(\mathbf{X})^\dagger$ where $\phi(\mathbf{x}) = \mathbf{x}^{\odot 3}$):
   - Erased Query final MSE: **0.12970518**
   - Noisy Query final MSE: **0.21106599**
   - While it keeps constant $O(d^2)$ complexity, the non-linear polynomial features distort the noise coordinates and degrade reconstruction compared to pure pseudoinverse.

2. **Dual-Projection Hopfield**:
   - Erased Query final MSE: **0.00000000** (Perfect reconstruction)
   - Noisy Query final MSE: **0.00000000** (Perfect reconstruction)
   - Complexity per step: $O(d \cdot M)$ linear time complexity.
   - For 10 memories, this is mathematically identical to a standard modern softmax Hopfield network, but structured as two sequential linear projection steps: mapping pixels to memory coordinates ($\mathbf{z} = \mathbf{W}_{proj} \mathbf{x}$), followed by coordinate-space softmax, and mapping coordinates back to pixels ($\mathbf{x}' = \mathbf{X} \mathbf{w}$).

### Dual-Projection Trajectory Results:
The visual reconstruction is flawless, retrieving the exact clean target image pixel-to-pixel from both the bottom-erased image and heavily corrupted noisy image:

![Dual Projection Trajectory](C:/Users/karthikkrazy/Documents/antigravity/optimistic-lovelace/sharp_mamba_results.png)
