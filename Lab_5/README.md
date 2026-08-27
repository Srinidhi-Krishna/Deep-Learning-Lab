# Lab 5 — Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation (MobileNetV2)
Course: CS3807 – Deep Learning Laboratory
Program: B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
Experiment 5

## Objective
Systematically study the effect of weight initialization, regularization, optimization algorithms, CNN hyperparameters, transfer learning, fine-tuning and 5-fold cross-validation on image classification performance, using a single lightweight CNN architecture (MobileNetV2) on the Oxford-IIIT Pet dataset.

## Dataset
Source: Oxford-IIIT Pet Dataset (direct download from robots.ox.ac.uk)
Total Images: 7,390
Training Images: 5,173
Validation Images: 1,108
Testing Images: 1,109
Classes: 37 pet breeds
Image Size: 224 x 224 x 3 (RGB, resized and MobileNetV2-preprocessed)
Missing values: A small number of corrupt JPEG files, automatically skipped by the tf.data decoding pipeline

## What's in this notebook
- Dataset Preparation — direct download and extraction, breed-label parsing from filenames, 70/15/15 train/val/test split, MobileNetV2 preprocessing pipeline
- Weight Initialization Study — Zero, Random, Xavier/Glorot and He initialization compared over 8-epoch runs on a frozen MobileNetV2 base
- Regularization Study — No Regularization, L2, Dropout and Batch Normalization compared over 8-epoch runs
- Batch Normalization — worked numerical example plus a With-BN vs. Without-BN comparison
- Optimizer Study — SGD, Momentum, RMSProp and Adam compared over 8-epoch runs
- Hyperparameter Tuning — learning rate, batch size and dropout rate swept independently over 5-epoch runs
- Transfer Learning and Fine-Tuning — frozen-base feature extraction vs. fine-tuning of block_15, block_16 and Conv_1 at a reduced learning rate
- 5-Fold Cross-Validation — four candidate configurations (C1–C4) evaluated across 5 folds to select the final configuration
- Final Model Evaluation — Accuracy, Precision, Recall, F1-score, Confusion Matrix on the untouched test set

## Results Summary
Metric: Test Accuracy | Value: 91.88%
Metric: Precision (weighted) | Value: 91.75%
Metric: Recall (weighted) | Value: 91.56%
Metric: F1-score (weighted) | Value: 91.49%
Metric: Best 5-Fold CV Configuration | Value: C3 (Dropout 0.5, No BatchNorm, RMSProp, lr=1e-3)
Metric: Mean CV Accuracy ± SD (C3) | Value: 90.62% ± 0.54
Metric: Best Optimizer (peak val. accuracy) | Value: Adam, 92.33% (converged by epoch 4)
Metric: Total Parameters (final model) | Value: 2,426,725
Metric: Training Time (final model) | Value: 73.46 s

Per-class notes: the confusion matrix is overwhelmingly diagonal across all 37 breeds, with only a small number of off-diagonal misclassifications scattered between individual visually-similar breed pairs rather than concentrated in any one region.

## Repository Structure
Lab5_CNNStudy/
├── README.md                     <- this file
├── DL_LAB_5.ipynb                 <- main notebook (all code + outputs)
├── report/
│   ├── DL_Lab_5.pdf               <- full pdf lab report
│   └── Lab_5.tex                  <- full latex code of lab report
└── figures/                       <- generated plots (.eps + .pdf), 600 DPI
    ├── Plot1_TrainingLoss_Initialization.eps / .pdf
    ├── Plot2_ValAccuracy_Initialization.eps / .pdf
    ├── Plot3_TrainVal_Accuracy.eps / .pdf
    ├── Plot4_TrainVal_Loss.eps / .pdf
    ├── Plot5_BatchNorm_Comparison.eps / .pdf
    ├── Plot6_TrainingLoss_Optimizers.eps / .pdf
    ├── Plot7_ValAccuracy_Optimizers.eps / .pdf
    ├── Plot8_LearningRate_ValAccuracy.eps / .pdf
    ├── Plot9_BatchSize_ValAccuracy.eps / .pdf
    ├── Plot10_Dropout_ValAccuracy.eps / .pdf
    ├── Plot11_FeatureExtraction_vs_FineTuning.eps / .pdf
    ├── Plot12_TrainVal_Loss_FineTuning.eps / .pdf
    ├── Plot13_KFold_CV_Accuracy.eps / .pdf
    └── Plot14_ConfusionMatrix.eps / .pdf

## How to Run
1. Open DL_LAB_5.ipynb in Kaggle Notebooks or Google Colab (GPU T4 x2 recommended for training speed).
2. Run all cells top to bottom — the Oxford-IIIT Pet dataset downloads automatically from robots.ox.ac.uk, and MobileNetV2 ImageNet weights download automatically via tensorflow.keras.applications, so no manual download is needed.
3. Generated figures are saved as .eps (600 DPI, required submission format) alongside a matching .pdf for fast LaTeX compilation.
4. On Kaggle, figures and outputs are saved under /kaggle/working/ and appear directly in the notebook's Output tab for download.

## Requirements
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow

## Report
The full lab report (theory, methodology, source code, results, and plot interpretations) is available in report/Lab_5.tex.
