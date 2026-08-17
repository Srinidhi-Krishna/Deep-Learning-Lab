Lab 4 — Transfer Learning and Fine-Tuning of a Pretrained CNN (VGG16)

Course: CS3807 – Deep Learning Laboratory
Program: B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
Experiment 4

Objective

Implement transfer learning using a VGG16 convolutional neural network pretrained on ImageNet, adapting it to CIFAR-10 via a feature-extraction phase followed by fine-tuning of the top convolutional block, and evaluate the impact of fine-tuning on classification performance.

Dataset

Source: CIFAR-10 (via tensorflow.keras.datasets.cifar10)
Training Images: 50,000
Testing Images: 10,000
Classes: 10 (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
Image Size: 32 × 32 × 3 (RGB)
Missing values: None

What's in this notebook

Dataset Preparation — pixel normalization to [0,1], one-hot encoding, sample image grid
Transfer Learning Model (VGG16) — pretrained ImageNet base, include_top=False, frozen convolutional layers, custom head: GlobalAveragePooling2D → Dense(256, ReLU) → Dense(10, Softmax)
Model Training — 15-epoch feature-extraction phase (Adam, lr=0.001, batch size 32) with frozen VGG16 base, only the classifier head trained
Fine-Tuning — block5 of VGG16 unfrozen, recompiled at lr=0.0001, trained for 10 further epochs jointly with the head
Combined Training Curve — accuracy before vs. after fine-tuning, with the fine-tuning start point marked
Evaluation — Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report
Misclassified Samples — 10 sample test images with true vs. predicted labels
Hyperparameter Study — 3 configurations (learning rate, batch size, optimizer, dense units) compared over 5-epoch runs

Results Summary

| Metric                  | VGG16 Transfer Learning (Fine-Tuned) |
|--------------------------|----------------------------------------|
| Test Accuracy            | 72.82%                                |
| Precision (weighted)     | 73.28%                                |
| Recall (weighted)        | 72.82%                                |
| F1-score (weighted)      | 72.77%                                |
| Total Parameters         | 14,848,586                            |
| Trainable (feature extraction) | 133,898                         |
| Trainable (after fine-tuning)  | 7,213,322                       |

Per-class notes: Ship (85% F1), Automobile (83% F1), and Frog (77% F1) were the best-classified categories; Cat had the lowest F1-score (54%), reflecting frequent confusion with visually similar animal classes (dog, deer).

Repository Structure

Lab4_TransferLearning/
├── README.md                     <- this file
├── DL_LAB_4.ipynb                 <- main notebook (all code + outputs)
├── report/
│   ├── DL_Lab_4.pdf               <- full pdf lab report
│   └── Lab_4.tex                  <- full latex code of lab report
└── figures/                       <- generated plots (.eps + .pdf), 600 DPI
    ├── fig1_sample_images.eps / .pdf
    ├── fig2_accuracy.eps / .pdf
    ├── fig3_loss.eps / .pdf
    ├── fig4_finetune_accuracy.eps / .pdf
    ├── fig5_confusion_matrix.eps / .pdf
    └── fig6_misclassified.eps / .pdf

How to Run

Open DL_LAB_4.ipynb in Google Colab or Jupyter (GPU runtime recommended for training speed).
Run all cells top to bottom — CIFAR-10 downloads automatically via tensorflow.keras.datasets, and VGG16 ImageNet weights download automatically via tensorflow.keras.applications, so no manual download is needed.
Generated figures are saved as .eps (600 DPI, required submission format) alongside a matching .pdf for fast LaTeX compilation.

Requirements

numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow

Report

The full lab report (theory, methodology, source code, results, and plot interpretations) is available in report/Lab_4.tex.
