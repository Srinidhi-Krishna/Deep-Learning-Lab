# Lab 8–9 — Effect of Attention Type and Attention Position in a Pretrained ResNet50

**Course:** CS3807 – Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
**Experiments 8, 9**

## Objective

Study how different attention mechanisms modify a pretrained convolutional network and how the **position at which attention is inserted** influences image-classification performance — using a single ImageNet-pretrained ResNet50 backbone throughout. Channel, spatial, sequential channel–spatial and lightweight self-attention are compared under one controlled protocol, together with attention position, multi-stage attention, attention order, spatial kernel size and three established mechanisms (ECA, CBAM, Coordinate Attention), with quantitative, complexity and qualitative (internal attention maps, Grad-CAM) analysis.

## Dataset

- **Primary source:** TensorFlow Flowers (`flower_photos`), 3,670 RGB images, downloaded directly from the TensorFlow server
- **Classes:** daisy, dandelion, roses, sunflowers, tulips
- **Input representation:** $x \in \mathbb{R}^{224 \times 224 \times 3}$
- **Split:** stratified 70 : 15 : 15 with seed 42 (2,569 train / 550 validation / 551 test), saved once in `split.json` and re-used for every experiment
- **Preprocessing:** resize to 224×224, then ImageNet `resnet50.preprocess_input`
- **Augmentation (training only, identical for all models):** horizontal flip, rotation (0.05), zoom (0.10), translation (0.10), contrast (0.10)

## What's in this notebook

- **Setup & Configuration** — seeds, batch size 32, two-stage transfer learning (5 epochs head-only at $10^{-3}$, 10 epochs with `conv5_x` unfrozen at $10^{-4}$, BatchNorm frozen), Adam / categorical cross-entropy, dropout 0.3
- **Attention Modules** — `ChannelAttention` (SE-style, $r=16$), `SpatialAttention` ($k\times k$), `ChannelSpatialAttention` (C→S or S→C, also CBAM), `SelfAttention` (non-local, learnable $\gamma$), `ECA`, `CoordAtt`; each exposes its internal attention values for visualisation
- **ResNet50 with Attention** — backbone split into `conv2_x … conv5_x` stage-models; attention inserted at $P_1$–$P_4$ (after conv2_x … conv5_x); input/output feature dimensions of every block recorded
- **Experiment 1: Attention Type** — baseline, channel, spatial, channel+spatial, self-attention, combined (at $P_3$)
- **Experiment 2: Attention Position** — channel→spatial at $P_1$, $P_2$, $P_3$, $P_4$
- **Experiment 3: Multiple Positions** — $P_3{+}P_4$ and $P_2{+}P_3{+}P_4$
- **Experiment 4: Attention Order** — channel→spatial vs. spatial→channel
- **Experiment 5: Spatial Kernel Size** — $k = 3, 5, 7$
- **Additional Mechanisms** — ECA, CBAM, Coordinate Attention at $P_3$
- **Quantitative & Complexity Analysis** — accuracy, macro precision/recall/F1, total/trainable/attention parameters, inference time per image, test loss
- **Graphical Analysis** — train/validation accuracy and loss curves, mechanism vs. accuracy / macro F1, accuracy vs. parameters, macro F1 vs. inference time
- **Confusion Matrices & Class-wise Analysis** — baseline vs. selected attention model, per-class F1 and gain over baseline
- **Qualitative Analysis** — feature maps before/after attention, channel-attention weights (top-20), spatial-attention maps and overlays, self-attention response, attention maps at $P_1$–$P_4$, Grad-CAM comparisons, and a correct/incorrect sample study
- **Export** — results, split, CSV tables, figures (EPS + PDF, 600 DPI) zipped for download

## Results Summary

17 models were trained; the main comparison at $P_3$ is below (test set: 551 images).

| Metric | Baseline | Channel (SE) | Spatial | Channel+Spatial | Self-Attention |
|---|---|---|---|---|---|
| Test Accuracy | 0.9328 | 0.9474 | **0.9583** | 0.9238 | 0.9328 |
| Macro F1 | 0.9331 | 0.9472 | **0.9583** | 0.9254 | 0.9326 |
| Attention Parameters | 0 | 132,160 | 99 | 132,259 | 1,310,721 |
| Total Parameters | 23,597,957 | 23,730,117 | 23,598,056 | 23,730,216 | 24,908,678 |
| Inference (ms/img) | 3.967 | 3.907 | 3.893 | 3.945 | 4.081 |

| Metric | Value |
|---|---|
| Highest Test Accuracy | Spatial attention at $P_3$ — 0.9583 (baseline 0.9328); lowest: C+S at $P_1$ — 0.9165 |
| Attention Position (C→S) | $P_1$ 0.9165, $P_2$ 0.9347, $P_3$ 0.9238, $P_4$ **0.9365** — no monotonic trend; parameter cost grows from 8,563 ($P_1$) to 526,563 ($P_4$) |
| Multiple Positions | $P_3{+}P_4$ 0.9328 (= baseline), $P_2{+}P_3{+}P_4$ 0.9201 — more positions did not help |
| Attention Order | Spatial→Channel 0.9474 vs. Channel→Spatial 0.9238 (same parameters) |
| Spatial Kernel Size | $3\times3$ 0.9474, $5\times5$ 0.9437, $7\times7$ **0.9583** — non-monotonic |
| Additional Mechanisms | ECA 0.9437 (5 parameters), CBAM 0.9328, Coordinate Attention 0.9238 |
| Best Channel–Spatial Model (used for qualitative study) | B4, C+S at $P_4$ — macro F1 0.9366 |
| Class-wise Effect | Tulip F1 improved for all 16 attention models; rose F1 decreased for 12 of 16 |
| Complexity | Overhead up to 6.1% parameters and 5.9% inference time (self-attention); light modules within timing noise |
| Internal Attention Maps | Spatial gates nearly uniform (≈1) with border effects; self-attention collapses to one hotspot per image ($\gamma = 0.0297$) — not object-aligned |
| Statistical Caveat | Single seed and 551 test images (1 image ≈ 0.18 pt, standard error ≈ 1 pt); baseline has the best validation accuracy (0.953), so no attention model is shown to beat it |

**Code and notebook:** https://github.com/Srinidhi-Krishna/Deep-Learning-Lab/tree/main/Lab_8_9
