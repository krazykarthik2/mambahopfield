# Findings: Latent Prototypes Memory Bank (Option B)

We implemented a prototype-based memory lookup model using a pre-trained Denoising Autoencoder (DAE) and evaluated its downstream classification performance under noisy inputs.

---

## Performance Summary (after 10 epochs)

| Configuration | Validation Loss | Validation Accuracy (%) |
| :--- | :---: | :---: |
| **Random Prototypes (Frozen)** | **1.0371** | **65.45%** |
| **Baseline (No Memory Lookup)** | 1.1072 | 62.25% |
| **Learned Class Prototypes** | 1.1842 | 61.55% |

### Performance Visualization

![Validation Curves for Prototype Experiment](C:/Users/karthikkrazy/.gemini/antigravity/brain/0241d110-ba49-44a7-8ea2-0adac73b6146/prototype_results.png)

---

## Mathematical and Structural Analysis: Why did this happen?

### 1. The ReLU Alignment Effect (Why "Memory" Performed Worst)
The DAE encoder uses a ReLU activation at the bottleneck:
$$\mathbf{z} = \text{ReLU}(\mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2)$$
Because of this, all elements of the latent vector $\mathbf{z}$ are strictly non-negative ($z_i \geq 0$). 
- When vectors are strictly non-negative, they all point in the same quadrant/orthant of the 32-dimensional vector space.
- As a result, the cosine similarity between any two latent vectors is always highly positive and very high (e.g., $0.80$ to $0.98$), regardless of the digit class.
- When we run softmax over these similarities:
  $$\mathbf{w} = \text{softmax}(\mathbf{s} / \tau)$$
  the attention weight distribution becomes almost uniform or highly saturated. The retrieved representation $\mathbf{z}_{retrieved}$ is nearly identical for every single digit. Adding this constant offset acts as a distractor, clogging the features and making classification harder.

### 2. Why "Random" Performed Best
The random prototype memory bank was initialized with Gaussian noise $\mathcal{N}(0, I)$ containing both positive and negative values:
- Unlike the ReLU-constrained latent space, the random vectors are distributed throughout the entire space.
- This creates clean, sharp cosine similarity distances.
- During training, the classification layer learns to use this random mapping as a **high-dimensional projection hashing scheme** (similar to kernel trick or random feature projection), which acts as a powerful regularizer, helping it achieve **65.45%** accuracy.
