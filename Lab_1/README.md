# Lab 1 — Single Layer Perceptron for Binary Classification

**Course:** CS3807 – Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
**Experiment 1**

## Objective

Implement a Single Layer Perceptron from scratch to understand the perceptron learning rule, the role of the step activation function, and how to evaluate a binary classifier on a real-world dataset — the [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication).

## Dataset

- **Source:** UCI Machine Learning Repository — Banknote Authentication
- **Instances:** 1372
- **Features:** Variance, Skewness, Curtosis, Entropy (all numerical, extracted from wavelet-transformed banknote images)
- **Target:** 0 = Authentic, 1 = Forged
- **Missing values:** None

## What's in this notebook

1. **Dataset Exploration** — shape, missing values, descriptive statistics, class balance
2. **Exploratory Data Analysis** — feature histograms, correlation heatmap, scatter plot, boxplots
3. **Preprocessing** — Min-Max normalization to [0, 1], 80/20 stratified train/test split
4. **Perceptron (from scratch)** — NumPy implementation with step activation and the classic perceptron learning rule
5. **Training** — 50-epoch training loop with per-epoch misclassification tracking
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix
7. **Learning Rate Study** — comparison across η = 0.001, 0.01, 0.1
8. **Optional Extension** — 2-feature decision boundary visualization, and a comparison against scikit-learn's `Perceptron`

## Results Summary

| Metric    | From-Scratch Perceptron | Scikit-learn Perceptron |
|-----------|:------------------------:|:-------------------------:|
| Accuracy  | 97.09%                  | 97.09%                  |
| Precision | 93.85%                  | 93.85%                  |
| Recall    | 100.00%                 | 100.00%                 |
| F1-score  | 96.83%                  | 96.83%                  |

The from-scratch implementation matches scikit-learn's reference implementation exactly under identical hyperparameters.

## Repository Structure

```
Lab1_Perceptron/
├── README.md                 <- this file
├── DL_LAB_1.ipynb             <- main notebook (all code + outputs)
├── report/
│   └── DL_Lab_1_Report.pdf    <- full pdf lab report
└── figures/                   <- generated plots (.eps + .pdf), 600 DPI
    ├── Fig1_Histograms.eps / .pdf
    ├── Fig2_CorrelationHeatmap.eps / .pdf
    ├── Fig3_ScatterPlot.eps / .pdf
    ├── Fig4_Boxplots.eps / .pdf
    ├── Fig5_ConvergenceCurve.eps / .pdf
    ├── Fig6_ConfusionMatrix.eps / .pdf
    ├── Fig7_WeightEvolution.eps / .pdf
    ├── Fig8_BiasEvolution.eps / .pdf
    ├── Fig9_LearningRateComparison.eps / .pdf
    └── Fig10_DecisionBoundary.eps / .pdf
```

## How to Run

1. Open `DL_LAB_1.ipynb` in Google Colab or Jupyter.
2. Run all cells top to bottom — the notebook downloads the dataset directly from the UCI repository, so no manual download is needed.
3. Generated figures are saved as `.eps` (600 DPI, required submission format) alongside a matching `.pdf` for fast LaTeX compilation.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Report

The full lab report (theory, methodology, source code, results, and plot interpretations) is available in [`report/DL_Lab1_Report.tex`](./report/DL_Lab1_Report.tex).
