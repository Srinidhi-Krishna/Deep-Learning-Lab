# Lab 6 — End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

**Course:** CS3807 – Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
**Experiment 6**

## Objective

Develop an end-to-end understanding of recurrent sequence learning by implementing and comparing Vanilla RNN, LSTM and GRU models under identical experimental conditions, introducing Backpropagation Through Time (BPTT) and the vanishing/exploding gradient problem, and extending the pipeline to CNN + recurrent video understanding and sequence-to-sequence learning.

## Dataset

- **Primary source:** UCI Human Activity Recognition Using Smartphones (raw inertial signals, not the pre-computed 561-feature vectors)
- **Activities:** WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING (6 classes)
- **Input representation:** $X \in \mathbb{R}^{N \times 128 \times 9}$ — 128 time steps × 9 channels (3-axis body acceleration, 3-axis gyroscope, 3-axis total acceleration)
- **Laboratory subset:** 2,470 class-balanced training windows (stratified subset), 642 windows from the official held-out test partition
- **Split:** 70% train / 15% validation / 15% test, normalized using training-set statistics only
- **Secondary datasets:** a small UCF101-style video subset (3 classes: Basketball, Biking, WalkingWithDog) for CNN + LSTM/GRU video understanding, and a synthetic 5-integer sequence-reversal dataset for the seq2seq task

## What's in this notebook

- **Dataset Preparation** — raw signal download and extraction, class-balanced subset selection, train/val/test split with training-only normalization
- **Temporal Visualization** — multi-channel, multi-activity sensor signal plots
- **BPTT Numerical Exercise** — manual 3-step hidden-state calculation cross-checked against a from-scratch NumPy implementation
- **RNN / LSTM / GRU Classification** — identical architecture (32-unit recurrent layer → Dropout(0.2) → Dense(16, ReLU) → Dense(6, softmax)), identical optimizer/batch size/epochs, compared on accuracy, precision, recall, F1, parameter count and training time
- **Confusion Matrix Analysis** — per-activity recognition rates and confusion pairs for all three architectures
- **Effect of Sequence Length** — RNN/LSTM/GRU re-trained and re-evaluated at T = 32, 64 and 128 time steps
- **Video Understanding (CNN + LSTM/GRU)** — frozen MobileNetV2 feature extraction (10 frames/video) feeding a CNN–LSTM and a CNN–GRU classifier
- **Sequence-to-Sequence Learning** — LSTM encoder–decoder trained on a synthetic integer-reversal task, evaluated with token accuracy and sequence accuracy
- **Additional Exercises** — recurrent-unit sweep (16 vs. 64 units) and Bidirectional vs. unidirectional LSTM comparison

## Results Summary

| Metric | RNN | LSTM | GRU |
|---|---|---|---|
| Test Accuracy | 75.70% | 85.36% | **91.12%** |
| Macro Precision | 75.78% | 88.34% | **90.93%** |
| Macro Recall | 75.12% | 85.19% | **90.99%** |
| Macro F1-score | 74.93% | 84.77% | **90.92%** |
| Parameters | 1,974 | 6,006 | 4,758 |
| Training Time | 40.19 s | 31.48 s | **29.40 s** |

| Metric | Value |
|---|---|
| Best HAR Model | GRU (32 units) — 91.12% accuracy, 90.92% macro F1 |
| Best Sequence Length | T = 64 (outperforms full T = 128 for all three architectures) |
| Video Understanding (CNN–GRU) | 16.67% accuracy, 9.52% F1 — synthetic-clip fallback used (UCF101 download failed via SSL error), pipeline validated end-to-end |
| CNN Feature Dimension | D = 1,280 (MobileNetV2, frozen, global-average-pooled) |
| Seq2Seq Token Accuracy | 100.00% |
| Seq2Seq Sequence Accuracy | 100.00% |
| Bidirectional vs. Unidirectional LSTM | 90.86% vs. 84.77% macro F1 (BiLSTM: 11,894 params) |


