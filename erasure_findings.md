# Findings: Reconstruction of Heavily Erased Images

We evaluated the associative memory capability of the **Centered Linear Hopfield Memory Bank** under extreme information erasure. 

We took 5 clean test images of clothing, completely erased (zeroed out) the **bottom 50% of the pixels**, and compared the reconstruction ability of a standard DAE against our memory-augmented DAE.

---

## Reconstruction Visual Results

Below is the side-by-side reconstruction quality grid:

![Erased Reconstruction Comparison Grid](C:/Users/karthikkrazy/.gemini/antigravity/brain/0241d110-ba49-44a7-8ea2-0adac73b6146/erased_reconstructions.png)

---

## Visual Observations and Insights

### 1. DAE Baseline (Failure to Inpaint/Complete)
- When the DAE baseline receives an image with the bottom half erased, its latent representation $\mathbf{z}$ only represents the remaining top half of the garment.
- As a result, the baseline DAE's reconstructions look cut-off or faded. It struggles to generate the missing parts of the clothing (e.g., the bottom hem of the coat, the legs of the trousers, or the heels of the boots).

### 2. Linear Hopfield (Successful Associative Retrieval)
- In the memory-augmented model, the query $\mathbf{z}$ retrieves a clean, prototypical class representation from the $32 \times 32$ memory matrix $\mathbf{S}$. 
- The retrieved representation acts as an associative prior. When added back to the latent state, the decoder is provided with a complete representation of a clothing archetype.
- As seen in the bottom row, the **Linear Hopfield** model successfully reconstructs the complete garment, drawing the bottoms of trousers, shoes, and coats that were completely missing in the input.

This visually proves that the Centered Linear Hopfield Memory Bank operates as a true associative memory, completing occluded and erased inputs by recalling class prototypes.
