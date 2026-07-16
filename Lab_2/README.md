# Lab 2 — Multi-Layer Perceptron for Multi-Class Image Classification

**Course:** CS3807 – Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
**Experiment 2**

## Objective

Implement a Multi-Layer Perceptron (MLP) using TensorFlow/Keras for multi-class image classification on the Fashion-MNIST dataset. Learn image preprocessing, model construction, training, evaluation, and automated hyperparameter optimization.

## Dataset

- **Source:** Fashion-MNIST (built into `keras.datasets`)
- **Training Images:** 60,000
- **Testing Images:** 10,000
- **Classes:** 10 — T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Image Size:** 28 × 28 (grayscale)
- **Missing values:** None

## What's in this notebook

- **Dataset Exploration** — shape, class distribution, sample image visualization
- **Preprocessing** — flattening (28×28 → 784), pixel normalization to [0, 1], one-hot label encoding
- **Model Construction** — baseline MLP: 784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
- **Training** — 20 epochs, batch size 32, Adam optimizer, categorical cross-entropy loss
- **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report
- **Hyperparameter Optimization** — manual Randomized Search with 5-fold cross-validation over hidden layers, neurons, learning rate, batch size, epochs, optimizer, activation, and dropout rate (re-implemented from scratch after a SciKeras/scikit-learn version incompatibility)
- **Model Comparison** — baseline vs. optimized MLP across all metrics

## Results Summary

| Metric | Baseline MLP | Optimized MLP |
|---|---|---|
| Accuracy | 87.60% | 88.46% |
| Precision | 87.80% | 88.67% |
| Recall | 87.60% | 88.46% |
| F1-score | 87.40% | 88.19% |
| Training Time | 86.93 s | 345.55 s |

**Best hyperparameters (Randomized Search, 5-fold CV):** 3 hidden layers, 256 neurons, learning rate 0.001, batch size 16, Adam optimizer, Sigmoid activation, 20 epochs, dropout rate 0.2 (CV accuracy: 85.61%).

The optimized model outperforms the baseline across all metrics at roughly 4× the training cost. Both models struggle most on the Shirt class, which is visually similar to T-shirt/top, Pullover, and Coat.

## Repository Structure

```
Lab2_MLP/
├── README.md                  <- this file
├── DL_LAB_2.ipynb              <- main notebook (all code + outputs)
├── report/
│   └── DL_Lab_2_Report.pdf     <- full pdf lab report
└── figures/                    <- generated plots (.eps + .pdf), 600 DPI
    ├── fig1_sample_images.eps / .pdf
    ├── fig2_class_distribution.eps / .pdf
    ├── fig3_train_accuracy.eps / .pdf
    ├── fig4_val_accuracy.eps / .pdf
    ├── fig5_train_loss.eps / .pdf
    ├── fig6_val_loss.eps / .pdf
    ├── fig7_confusion_matrix_baseline.eps / .pdf
    ├── fig7b_confusion_matrix_optimized.eps / .pdf
    ├── fig8_hyperparam_search_results.eps / .pdf
    └── fig9_model_comparison.eps / .pdf
```

## How to Run

1. Open `DL_LAB_2.ipynb` in Google Colab or Jupyter.
2. Run all cells top to bottom — Fashion-MNIST loads directly via `keras.datasets.fashion_mnist`, so no manual download is needed.
3. Generated figures are saved as `.eps` (600 DPI, required submission format) alongside a matching `.pdf` for fast LaTeX compilation.

## Requirements

```
numpy
pandas
tensorflow
matplotlib
seaborn
scikit-learn
```

## Report

The full lab report (theory, methodology, source code, hyperparameter optimization, results, and plot interpretations) is available in `report/DL_Lab_2_Report.tex`.
