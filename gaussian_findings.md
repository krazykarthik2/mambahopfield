# Findings: Gaussian Start Nodes for Middle-Input Networks

We tested the hypothesis that feeding a flat constant vector of ones ($\mathbf{1}$) at the start nodes of the pre-layers limits the model's capacity, and that using Gaussian-distributed start nodes could improve performance.

---

## Experimental Setup

The model architecture was identical to the first experiment, but we compared three ways of feeding the start nodes of the pre-input layers:
1. **ones**: Flat vector of ones (`torch.ones(1, 128)`), constant for all samples.
2. **fixed_gaussian**: A single vector sampled from $\mathcal{N}(0, I)$ at initialization, constant for all samples.
3. **sample_gaussian**: A fresh vector sampled from $\mathcal{N}(0, I)$ for each individual sample in the batch at every forward pass.

---

## Performance Summary (after 20 epochs on MNIST)

| Configuration | Validation Loss | Validation Accuracy (%) | Peak Validation Acc (%) |
| :--- | :---: | :---: | :---: |
| **Ones (Constant)** | **0.2660** | **93.00%** | **93.65% (Epoch 19)** |
| **Sample Gaussian (Stochastic)** | 0.2817 | 92.60% | 93.15% (Epoch 15) |
| **Fixed Gaussian (Constant)** | 0.3294 | 92.15% | 93.30% (Epoch 17) |

### Performance Visualization

![Validation Loss and Accuracy for Gaussian Start Nodes](C:/Users/karthikkrazy/.gemini/antigravity/brain/0241d110-ba49-44a7-8ea2-0adac73b6146/gaussian_results.png)

---

## Key Insights

### 1. Training Convergence and Optimization Speed
Early in training (e.g. Epoch 5), the **Sample Gaussian** model has lower validation loss and generalizes slightly better because of the regularizing noise. 
However, over 20 epochs, the **Ones** model converges to a slightly higher validation accuracy (**93.00%**) and lower validation loss (**0.2660**). 
This is because adding stochastic noise (Sample Gaussian) increases training difficulty (Train Loss at epoch 20 is 0.0439 vs. 0.0156 for Ones), slowing down optimization convergence.

### 2. Information Capacity of Constant Vectors
The **Fixed Gaussian** configuration underperformed both Ones and Sample Gaussian, ending with **92.15%** validation accuracy and a higher validation loss (**0.3294**). 
Since both Ones and Fixed Gaussian feed a static/constant vector, they have the same theoretical information carrying capacity. However, a vector of all ones provides a clean, uniform starting bias, whereas a fixed random Gaussian vector introduces arbitrary initial scale differences that can distort gradient propagation or require extra optimization steps to correct.

