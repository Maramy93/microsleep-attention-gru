# Attention-Based Temporal Resolution Enhancement for Microsleep Forecasting

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

Official implementation of our Attention-GRU architecture for **early microsleep forecasting** from EEG.

Unlike conventional recurrent models, the proposed model introduces a **causal multi-head self-attention layer before the GRU** to preserve fine-grained temporal information when forecasting microsleep several seconds before onset.

---

# Motivation

Predicting microsleep **at onset (0 seconds)** is relatively straightforward because the EEG already contains strong microsleep signatures.

However, forecasting **1–2 seconds before onset** is substantially more difficult.

As the prediction horizon increases:

- transient EEG changes become weaker
- temporal resolution decreases
- discriminative information becomes blurred
- sensitivity drops dramatically

Traditional recurrent models summarize EEG over time using hidden states.

Although this is effective for sequence modelling, it may smooth away subtle transient neural activity that is critical for **early warning**.

Our hypothesis is:

> Important EEG events still exist before microsleep onset, but they become diluted when represented only by recurrent hidden states.

Therefore, we introduce a **causal multi-head self-attention layer** before the GRU to selectively preserve informative temporal patterns before sequence modelling.

The proposed architecture enables the model to focus on informative EEG segments while respecting temporal causality, leading to improved forecasting performance across multiple prediction horizons. 

---

# Repository Structure

```
microsleep-attention-gru
│
├── README.md
├── requirements.txt
│
├── configs
│   └── default.yaml
│
├── src
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   ├── metrics.py
│   ├── train.py
│   └── visualize.py
│
└── 
```

---

# Code Overview

## preprocessing.py

Performs EEG preprocessing including

- bandpass filtering
- epoch segmentation
- normalization
- preparation of training windows

---

## features.py

Extracts handcrafted EEG features including

- frequency band powers
- spectral entropy
- nonlinear complexity measures
- theta/(alpha+beta) ratio
- Hjorth parameters

These features form the temporal sequence presented to the neural network. :contentReference[oaicite:2]{index=2}

---

## model.py

Implements the proposed architecture.

Pipeline:

```
EEG Features
      │
      ▼
Causal Multi-head Attention
      │
      ▼
2-layer GRU
      │
      ▼
Fully Connected Layer
      │
      ▼
Sigmoid
```

The causal attention layer prevents future information leakage while allowing the model to emphasize informative temporal regions before recurrent processing. 

---

## train.py

Training pipeline

Includes

- train loop
- validation
- early stopping
- Adam optimizer
- focal loss
- model checkpointing

---

## metrics.py

Computes

- Accuracy
- Sensitivity
- Specificity
- Geometric Mean
- ROC AUC
- PR AUC

---

## visualize.py

Creates

- ROC curves
- Precision-Recall curves
- EEG plots
- Attention heatmaps
- Feature visualizations

Check the paper below
---

# Dataset

This work uses the publicly available

**Maintenance of Wakefulness Test (MWT) Dataset**

https://zenodo.org/records/3251716

The dataset contains

- 76 subjects
- multi-channel EEG
- EOG
- expert-labelled Wake
- Drowsiness
- Microsleep episodes

Signals were segmented into 1-second windows for forecasting experiments. 

---

# Model

Our proposed model extends a conventional GRU by introducing

✓ causal multi-head attention

✓ focal loss

to improve long-horizon forecasting.

Compared with a vanilla GRU, attention preserves informative temporal patterns that would otherwise be smoothed by recurrent processing.

---

# Results

Across prediction horizons

| Horizon | Vanilla GRU | Attention GRU |
|----------|------------|---------------|
| 0 s | Strong | Better |
| 1 s | Performance drops | Maintained |
| 2 s | Significant degradation | Stable performance |

The largest improvements were observed in

- Sensitivity
- AUC-ROC
- AUC-PR

particularly at extended forecasting horizons where temporal degradation becomes most severe. 

---

# Figures

## Overall framework

Check the paper below

---

# Citation

If you use this repository please cite

```
@article{Mohamed2025AttentionGRU,
  title={Attention-Based Temporal Resolution Enhancement for Microsleep Forecasting Across Multiple Horizons},
  author={Mohamed, Maram and O'Connor, Noel and Redmond, Peter},
  year={2025}
}
```

paper link : https://zenodo.org/records/19184602

---

# Acknowledgements

Dataset

Maintenance of Wakefulness Test (MWT)

https://zenodo.org/records/3251716

---

# License

MIT
