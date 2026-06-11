# 🌄 HimalayaGuard - Capstone Project
### *Know What Awaits Ahead*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg" />
  <img src="https://img.shields.io/badge/NumPy-Linear%20Algebra-green.svg" />
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-success.svg" />
</p>

---

## Overview

This is a multi-modal machine learning framework designed to transform raw trekking data into actionable route intelligence.

Using historical Himalayan trek data, the system learns latent terrain structure, groups routes with similar characteristics, and predicts operational costs through a fused-feature deep learning pipeline.

Rather than viewing a trek as a collection of disconnected attributes, HimalayaGuard builds a compact representation of the journey itself—capturing how duration, altitude, and difficulty interact across routes.

> **Know what awaits ahead before taking the first step.**

---

## ✨ Features

### Data Engineering
- Robust preprocessing and feature extraction
- Cost normalization and parsing
- Duration and altitude extraction
- Missing value handling and validation

### Custom Linear Algebra Pipeline
- Manual Singular Value Decomposition (SVD)
- Terrain-space projection
- Principal component analysis without high-level abstractions
- Variance contribution analysis

### Himalayan Terrain Space
Routes are projected into a latent geometric space where similar treks naturally appear closer together.

This representation captures hidden structural relationships between:

- Trek duration
- Maximum altitude
- Route difficulty

### Feature Fusion
Combines:

- Terrain embeddings (PC1, PC2)
- One-hot encoded trip grades

into a unified machine learning representation.

### Deep Learning Cost Prediction
PyTorch-based regression model that learns:

```
Terrain Characteristics
          +
Difficulty Profile
          ↓
Estimated Trek Cost
```

---

## 🏗 Architecture

```text
┌─────────────────────┐
│ Trek Data.csv       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Data Engineering    │
│ Feature Extraction  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Standardization     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Custom SVD Engine   │
│ Terrain Projection  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Fusion      │
│ PC1 + PC2 + Grades  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ PyTorch Regressor   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Cost Estimation     │
└─────────────────────┘
```

---

## Core Methodology

### 1. Route Feature Engineering
Raw trekking metadata is converted into structured numerical representations:

- Cost (USD)
- Duration (Days)
- Maximum Altitude (m)
- Trip Grade

### 2. Terrain Space Construction

After standardization, Singular Value Decomposition is applied:

```math
X = U \Sigma V^T
```

The leading singular vectors define a low-dimensional terrain space that preserves dominant route variation.

### 3. Latent Route Representation

Each trek receives:

- PC1 coordinate
- PC2 coordinate

representing its location within the learned terrain manifold.

### 4. Neural Cost Regression

Fused features are passed into a PyTorch network trained with:

- Mean Squared Error Loss
- Adam Optimization
- Automatic Differentiation

---

## 🚀 Running the Project

```bash
uv sync
uv run python main.py
```

or launch the notebook:

```bash
jupyter notebook main.ipynb
```

---

## Outputs

The project produces:

- Clean engineered trekking dataset
- Terrain-space projections
- Variance analysis tables
- Route embeddings
- Neural-network training curves
- Cost predictions

---