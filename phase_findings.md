# Findings: Phase-Based Training and Frozen Memory Banks

We tested a phase-based training strategy: pre-training the pre-input layers ("memory bank") to learn representation priors, then freezing them, and training the rest of the network (input projection and post-layers) from scratch.

---

## Experimental Setup

The experiment had two phases:
1. **Phase 1 (Pre-training)**: Train a model (`trainable_pre`) for 10 epochs. Extract the weights of the learned pre-layers.
2. **Phase 2 (Evaluation)**: Train three new model configurations for 10 epochs on MNIST:
   - **frozen_learned_pre**: The pre-layers are loaded with the learned weights from Phase 1 and frozen.
   - **frozen_random_pre**: The pre-layers are initialized randomly and frozen.
   - **trainable_pre**: The pre-layers are trainable from scratch.

---

## Performance Summary (after 10 epochs on MNIST)

| Configuration | Validation Loss | Validation Accuracy (%) |
| :--- | :---: | :---: |
| **Frozen Random Pre-layers** | **0.2477** | **93.20%** |
| **Fully Trainable Pre-layers** | 0.2505 | 92.45% |
| **Frozen Learned Pre-layers** | 0.2804 | 92.00% |

### Performance Visualization

![Validation Curves for Phase Experiment](C:/Users/karthikkrazy/.gemini/antigravity/brain/0241d110-ba49-44a7-8ea2-0adac73b6146/phase_results.png)

---

## Key Insights and Interpretation

### 1. The Co-Adaptation Problem
The **Frozen Learned** model performed the worst (**92.00%** validation accuracy).
* **Explanation**: In neural networks, layers co-adapt to each other during backpropagation. The pre-layers trained in Phase 1 learned a bias vector that was highly optimized for the *specific* features discovered by the initial model's input projection and post-layers.
* When we re-initialize the input projection and post-layers from scratch, the frozen pre-layers act as a rigid, complex bias constraint. Because the new layers are randomly initialized, they struggle to align their new feature representations with this rigid, pre-existing learned prior, hindering optimization.

### 2. Regularization in Frozen Random Pre-layers
Surprisingly, **Frozen Random** performed the best (**93.20%**).
* **Explanation**: By freezing the random pre-layers, we fix a constant random bias offset in the middle of the network. Because these weights are frozen, they reduce the number of active parameter degrees of freedom. On a smaller subset of MNIST (10,000 images), this acts as a strong constraint/regularizer, reducing overfitting and allowing the input projection and post-layers to generalize better.

### 3. Conclusion on Phase-Based Memory Training
Freezing the memory bank and re-initializing everything else disrupts co-adapted features. 
* **Design Rule**: If you want to use a pre-trained memory bank (pre-layers), **do not train the downstream layers completely from scratch**. Instead, keep the pre-trained downstream layers and fine-tune them, or keep the pre-layers trainable with a lower learning rate (discriminative learning rates) rather than locking them completely and starting other layers from scratch.
