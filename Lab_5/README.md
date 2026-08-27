# Lab 5 — Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation (MobileNetV2)

**Course:** CS3807 – Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
**Experiment 5**

## Objective

Systematically study the effect of weight initialization, regularization, optimization algorithms, CNN hyperparameters, transfer learning, fine-tuning and 5-fold cross-validation on image classification performance, using a single lightweight CNN architecture (MobileNetV2) on the Oxford-IIIT Pet dataset.

## Dataset

- **Source:** Oxford-IIIT Pet Dataset (direct download from robots.ox.ac.uk)
- **Total Images:** 7,390
- **Training Images:** 5,173
- **Validation Images:** 1,108
- **Testing Images:** 1,109
- **Classes:** 37 pet breeds
- **Image Size:** 224 × 224 × 3 (RGB, resized and MobileNetV2-preprocessed)
- **Missing values:** A small number of corrupt JPEG files, automatically skipped by the tf.data decoding pipeline

## What's in this notebook

- **Dataset Preparation** — direct download and extraction, breed-label parsing from filenames, 70/15/15 train/val/test split, MobileNetV2 preprocessing pipeline
- **Weight Initialization Study** — Zero, Random, Xavier/Glorot and He initialization compared over 8-epoch runs on a frozen MobileNetV2 base
- **Regularization Study** — No Regularization, L2, Dropout and Batch Normalization compared over 8-epoch runs
- **Batch Normalization** — worked numerical example plus a With-BN vs. Without-BN comparison
- **Optimizer Study** — SGD, Momentum, RMSProp and Adam compared over 8-epoch runs
- **Hyperparameter Tuning** — learning rate, batch size and dropout rate swept independently over 5-epoch runs
- **Transfer Learning and Fine-Tuning** — frozen-base feature extraction vs. fine-tuning of block_15, block_16 and Conv_1 at a reduced learning rate
- **5-Fold Cross-Validation** — four candidate configurations (C1–C4) evaluated across 5 folds to select the final configuration
- **Final Model Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix on the untouched test set

## Results Summary

| Metric | Value |
|---|---|
| Test Accuracy | 91.88% |
| Precision (weighted) | 91.75% |
| Recall (weighted) | 91.56% |
| F1-score (weighted) | 91.49% |
| Best 5-Fold CV Configuration | C3 (Dropout 0.5, No BatchNorm, RMSProp, lr=1e-3) |
| Mean CV Accuracy ± SD (C3) | 90.62% ± 0.54 |
| Best Optimizer (peak val. accuracy) | Adam, 92.33% (converged by epoch 4) |
| Total Parameters (final model) | 2,426,725 |
| Training Time (final model) | 73.46 s |

**Per-class notes:** the confusion matrix is overwhelmingly diagonal across all 37 breeds, with only a small number of off-diagonal misclassifications scattered between individual visually-similar breed pairs rather than concentrated in any one region.

## Repository Structure
