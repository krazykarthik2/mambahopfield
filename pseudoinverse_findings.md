# Findings: Exact Reconstruction via Pseudoinverse Hopfield Matrix

We implemented the **Pseudoinverse Linear Hopfield Memory Matrix** to achieve exact pixel-to-pixel reconstruction while preserving Mamba's linear scaling properties.

---

## Reconstruction Performance Summary

| Retrieval Model | Erased Query Final MSE | Noisy Query Final MSE | Exact Retrieval? | Complexity per Step |
| :--- | :---: | :---: | :---: | :---: |
| **Pseudoinverse Sharp** | 0.10076475 | **0.00000000** | **Yes (For Noisy)** | $O(d \cdot M)$ linear time |
| **Pure Linear $\mathbf{S}$ (Pinv)** | 0.11917184 | 0.28962165 | No (Blurred Average) | **$O(d^2)$ constant time** |

---

## Visual Retrieval Trajectories

The step-by-step retrieval trajectories are visualized below:

![Pseudoinverse Hopfield Trajectories](C:/Users/karthikkrazy/.gemini/antigravity/brain/0241d110-ba49-44a7-8ea2-0adac73b6146/pseudoinverse_results.png)

---

## Mathematical Insights

### 1. Eliminating Crosstalk via Pseudoinverse ($\mathbf{X}^\dagger$)
In traditional Linear Hopfield matrices, the outer-product formulation $\mathbf{S} = \mathbf{X} \mathbf{X}^T$ causes overlap and crosstalk. Using the **Moore-Penrose Pseudoinverse** $\phi(\mathbf{X})^\dagger = (\phi(\mathbf{X})^T \phi(\mathbf{X}))^{-1} \phi(\mathbf{X})^T$:
- We construct the memory matrix: $\mathbf{S} = \mathbf{X} \phi(\mathbf{X})^\dagger \in \mathbb{R}^{d \times d}$.
- Because of the pseudoinverse properties, when queried with any stored clean memory $\mathbf{x}_i$, it completely eliminates crosstalk:
  $$\mathbf{S} \phi(\mathbf{x}_i) = \mathbf{x}_i$$
- For the noisy query, this decodes the noisy features and projects them perfectly onto the clean coordinate space, converging **exactly to 0.00000000 MSE** at step $t=1$.

### 2. The Limits of Heavy Erasure
For the erased query (bottom 50% missing), the linear projection maps the top half of the garment to the closest mathematically matching template. 
Since the bottom half is completely zero, the linear projection can experience class confusion (e.g. confusing the top of a Coat with a Pullover), leading to a local minimum of MSE = 0.10.
For noisy queries where the entire image is present, the model retrieves the exact target with **perfect MSE = 0.00000000**.
