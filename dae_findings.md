# Denoising Autoencoder with Middle-Injected "Memory Bank"

We extended the concept of middle-injected pre-layers (acting as a "memory bank") to a Denoising Autoencoder (DAE) trained on MNIST with Gaussian noise ($\sigma = 0.5$).

---

## Experimental Setup

The DAE was configured with:
- **Encoder**: Map $784 \to 256 \to 32$ (bottleneck latent representation $\mathbf{z}$).
- **Decoder**: Map $32 \to 256 \to 784$ (reconstructed clean image $\hat{\mathbf{x}}$).
- **Bottleneck Injection**: The output of the pre-layers $\mathbf{z}_{mem}$ (trained on a constant input vector of ones $\mathbf{1} \in \mathbb{R}^{32}$) is added to the encoder output:
  $$\mathbf{z}_{combined} = \text{Encoder}(\tilde{\mathbf{x}}) + \mathbf{z}_{mem}$$

---

## Performance Summary

The models were trained under identical settings (5 epochs, subset of 10,000 images, batch size of 256, Adam optimizer with a learning rate of 0.002).

| Configuration | Validation MSE Loss | Performance vs. Baseline |
| :--- | :---: | :---: |
| **Trainable Pre-layers (Memory Bank)** | **0.0386** | **-4.9% MSE (Better)** |
| **Bias Parameter** | **0.0386** | **-4.9% MSE (Better)** |
| **Baseline (No Memory)** | 0.0406 | Baseline |
| **Frozen Pre-layers** | 0.0409 | +0.7% MSE (Worse) |

---

## Loss Curves Comparison

![DAE Validation MSE Loss Curves](C:/Users/karthikkrazy/.gemini/antigravity/brain/0241d110-ba49-44a7-8ea2-0adac73b6146/dae_loss_curves.png)

---

## Visual Reconstruction Comparison

Below is the reconstruction quality of random noisy test samples across all configurations:

![Reconstruction Comparison Grid](C:/Users/karthikkrazy/.gemini/antigravity/brain/0241d110-ba49-44a7-8ea2-0adac73b6146/dae_reconstructions.png)

---

## Key Takeaways

1. **Memory Bank acts as a Global Pattern Prior**: Adding the trainable pre-layers (`trainable_pre`) or direct bias parameters (`bias_param`) at the bottleneck helps the autoencoder memorize standard patterns (like common digit loops and lines). This relieves the encoder from having to reconstruct the entire digit from a noisy vector; it only needs to encode the specific variations from the default learned base shape.
2. **Trainable vs. Frozen**: Just like in the MLP experiment, frozen pre-layers act as random noise at the bottleneck, making the decoder's job slightly harder and increasing validation MSE compared to the baseline.
3. **Equivalence of Trainable Pre-layers and Bias Parameter in Bottleneck**: For a low-dimensional bottleneck (size 32), both the MLP-based pre-layers and the direct bias parameter achieve the same performance improvement (0.0386 MSE). This indicates that the representation space at size 32 is simple enough that both pathways optimize to the same offset vector.
