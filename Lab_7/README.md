# Lab 7 — End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders

**Course:** CS3807 – Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
**Experiment 7**

## Objective

Develop an end-to-end understanding of autoencoders and their variants for image representation, reconstruction, denoising and generative modeling — starting from a fully connected autoencoder, progressing to a Convolutional Autoencoder (CAE), introducing image corruption for a denoising autoencoder, and finally building a Variational Autoencoder (VAE), while comparing all models on reconstruction quality, latent-space structure and generative capability.

## Dataset

- **Primary source:** MNIST Handwritten Digit Dataset (grayscale, 28×28×1)
- **Classes:** digits 0–9 (labels used only for latent-space visualization, never as training targets)
- **Input representation:** $x \in \mathbb{R}^{784}$ (flattened) for the fully connected autoencoder, $x \in \mathbb{R}^{28 \times 28 \times 1}$ (spatial) for the CAE, denoising CAE and VAE
- **Laboratory subset:** 10,000 training images, 2,000 test images
- **Preprocessing:** pixel values normalized from $[0,255] \to [0,1]$
- **Reconstruction target:** the original image itself ($x \to \text{Encoder} \to z \to \text{Decoder} \to \hat{x}$); no labels required
- **Noise corruption (for denoising):** Gaussian noise ($\sigma \in \{0.1, 0.2, 0.3\}$) and salt-and-pepper noise ($p \in \{0.05, 0.10, 0.20\}$), clipped to $[0,1]$

## What's in this notebook

- **Fully Connected Autoencoder** — 784 → 128 → 32 → 16 (latent) → 32 → 128 → 784, ReLU hidden layers, sigmoid output, trained with Adam / BCE loss
- **Reconstruction Metrics** — MSE, MAE and SSIM computed on the test set for every model
- **Convolutional Autoencoder** — Conv2D/MaxPooling2D encoder, UpSampling2D/Conv2D decoder, compared against the FC-AE on accuracy and parameter efficiency
- **Denoising Convolutional Autoencoder** — trained on noisy input / clean target pairs; evaluated across multiple Gaussian and salt-and-pepper corruption levels
- **Variational Autoencoder** — 2-D latent space, reparameterization trick, joint reconstruction + KL-divergence loss
- **Latent-Space Analysis** — 2-D VAE latent-space visualization by digit label, random generation from the prior $\mathcal{N}(0,I)$, and linear latent-space interpolation between two digits
- **Reconstruction-Error Analysis** — per-image error histogram and inspection of the five highest-error test images (Convolutional Autoencoder)
- **Latent-Dimension Study** — Fully Connected Autoencoder re-trained and re-evaluated at $d_z = 2, 8, 16, 32$
- **Additional Exercises** — CAE latent-channel depth sweep (4/8/16/32), UpSampling2D vs. Conv2DTranspose decoder comparison, VAE latent dimension 2 vs. 8, Beta-VAE KL-weight sweep ($\beta = 1, 5, 10$), and a larger (100-sample) VAE generation/diversity study

## Results Summary

| Metric | FC Autoencoder | Convolutional AE | Denoising CAE | VAE |
|---|---|---|---|---|
| Test MSE | 0.019471 | **0.002721** | 0.004531 | 0.044798 |
| Test MAE | 0.053651 | **0.015880** | 0.020987 | 0.104901 |
| Mean SSIM | 0.7792 | **0.9717** | 0.9487 | 0.5051 |
| Parameters | 211,040 | 74,497 | 74,497 | 134,165 |
| Training Time | 30.04 s | 32.08 s | 19.88 s | 37.04 s |

| Metric | Value |
|---|---|
| Best Reconstruction Model | Convolutional Autoencoder — MSE 0.002721, SSIM 0.9717, only 74,497 parameters |
| VAE Reconstruction / KL / Total Loss | 150.5684 / 6.4880 / 157.0564 |
| Best FC-AE Latent Dimension | $d_z = 32$ (MSE 0.020833, SSIM 0.7603) — diminishing returns beyond $d_z = 16$ |
| Best CAE Latent Channel Depth (Exercise 1) | 32 channels — MSE 0.003382, SSIM 0.9642 |
| Best CAE Decoder (Exercise 4) | Conv2DTranspose — MSE 0.002193, SSIM 0.9768 (vs. UpSampling2D: 0.002721, 0.9717) |
| VAE Latent Dim 2 vs. 8 (Exercise 5) | 0.5051 vs. **0.1782** macro SSIM — posterior collapse observed at $d_z = 8$ |
| Best Beta-VAE Setting (Exercise 6) | $\beta = 1$ — MSE 0.043746, SSIM 0.5162 (higher $\beta$ trades fidelity for KL regularization) |
| VAE Generated-Sample Diversity (Exercise 7) | Mean pairwise pixel distance 6.1334 over 100 samples — mild mode collapse toward loop-shaped digits |

**Code and notebook:** https://github.com/Srinidhi-Krishna/Deep-Learning-Lab/tree/main/Lab_7
