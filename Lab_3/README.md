# Lab 3 — Convolutional Neural Networks for Image Classification

**Course:** CS3807 – Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
**Experiment 3**

## Objective

Implement and train a Convolutional Neural Network from first principles using PyTorch, to understand the convolution operation, stride/padding effects, pooling strategies, feature map visualization, and end-to-end image classification on a real-world dataset — CIFAR-10.

## Dataset

- **Source:** CIFAR-10 (via `torchvision.datasets.CIFAR10`)
- **Training Images:** 50,000
- **Testing Images:** 10,000
- **Classes:** 10 (Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
- **Image Size:** 32 × 32 × 3 (RGB)
- **Missing values:** None

## What's in this notebook

- **Dataset Exploration** — sample image grid, dataset dimensions, class distribution (5,000 images/class, perfectly balanced)
- **Convolution Kernel Study** — feature map comparison across 3×3 (→30×30), 5×5 (→28×28), 7×7 (→26×26) kernels, verified against the standard output-size formula
- **Stride & Padding Study** — 4 configurations (valid/same padding × stride 1/2) with formula vs. actual output-size verification
- **Feature Map Visualization** — first-layer activations across 8 of 16 channels
- **Pooling Comparison** — Max Pooling vs. Average Pooling, both downsampling 32×32 → 16×16
- **CNN Architecture** — Conv(3→16) → ReLU → MaxPool → Conv(16→32) → ReLU → MaxPool → Flatten → Dense(128) → Softmax(10), 268,650 trainable parameters
- **Training** — 20-epoch training loop (Adam optimizer, lr=0.001, batch size 32) with train/val accuracy & loss tracking
- **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report

## Results Summary

| Metric | Baseline CNN (PyTorch) |
|---|---|
| Test Accuracy | 66.73% |
| Precision (macro) | 66.53% |
| Recall (macro) | 66.73% |
| F1-score (macro) | 66.49% |
| Trainable Parameters | 268,650 |

**Per-class notes:** Car (78% F1), Ship (79% F1), and Frog (73% F1) were the best-classified categories; Cat had the lowest F1-score (46%), reflecting frequent confusion with visually similar animal classes (dog, deer, bird).

## Repository Structure

```
Lab3_CNN/
├── README.md                 <- this file
├── DL_LAB_3.ipynb             <- main notebook (all code + outputs)
├── report/
│   ├── DL_Lab_3.pdf           <- full pdf lab report
│   └── Lab_3.tex              <- full latex code of lab report
└── figures/                   <- generated plots (.eps + .pdf), 600 DPI
    ├── sample_images.eps / .pdf
    ├── class_distribution.eps / .pdf
    ├── kernel_comparison.eps / .pdf
    ├── feature_maps_conv1.eps / .pdf
    ├── pooling_comparison.eps / .pdf
    ├── training_accuracy.eps / .pdf
    ├── validation_accuracy.eps / .pdf
    ├── training_loss.eps / .pdf
    ├── validation_loss.eps / .pdf
    └── confusion_matrix.eps / .pdf
```

## How to Run

1. Open `DL_LAB_3.ipynb` in Google Colab or Jupyter (GPU runtime recommended for training speed).
2. Run all cells top to bottom — CIFAR-10 downloads automatically via `torchvision`, so no manual download is needed.
3. Generated figures are saved as `.eps` (600 DPI, required submission format) alongside a matching `.pdf` for fast LaTeX compilation.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
torchvision
```

## Report

The full lab report (theory, methodology, source code, results, and plot interpretations) is available in `report/Lab_3.tex`.
