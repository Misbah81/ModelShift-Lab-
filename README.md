# ModelShift Lab

### Interactive ML Robustness & Failure Analysis Platform

**ModelShift Lab** is an interactive machine-learning laboratory for exploring how classification models behave when the distribution of test data changes.

Instead of relying on a single accuracy score, the platform provides controlled experiments for comparing model performance, visualizing decision regions, and investigating how distribution shifts introduce prediction failures.

> **Measure model behaviour. Don't assume robustness from clean-data accuracy alone.**

---

## ✦ What ModelShift Lab Does

### 🧪 Controlled ML Experiments

Experiment with three synthetic binary classification datasets:

- **Two Moons**
- **Concentric Circles**
- **Class Imbalance**

Control dataset size, noise, imbalance, shift magnitude, and random seed.

### 🤖 Compare Multiple Models

Evaluate four scikit-learn classifiers under the same experimental conditions:

- Logistic Regression
- RBF SVM
- Random Forest
- MLP Neural Network

### 📈 Visualize Decision Regions

The platform generates prediction regions directly from the selected trained model over a two-dimensional feature grid.

You can visually inspect how different classifiers separate the same data and where their predictions fail.

### 🔄 Clean vs Shifted Data

Models are trained on the original training distribution and evaluated on:

**Clean test data → Shifted test data**

The test samples remain the same while their features are deterministically shifted, allowing the experiment to isolate the effect of the selected distribution shift.

### 🔍 Failure Analysis

Go beyond accuracy with:

- Accuracy
- Precision
- Recall
- F1 score
- Clean confusion matrix
- Shifted confusion matrix
- Clean errors
- Shifted errors
- Newly introduced errors
- Recovered predictions

This makes it possible to investigate **which failures appear after the distribution changes**.

### ♻️ Reproducible Experiments

Configurable random seeds allow experiments to be repeated using the same configuration.

Recent experiments can also be stored in browser-local history and reloaded for comparison.

### 📄 Experiment Reports

Export experiment results as CSV, including model metrics, confusion matrices, experiment settings, and failing test samples.

---

## 🧠 The Experiment

A typical ModelShift Lab experiment follows this process:

```text
Synthetic Dataset
       ↓
Training Split
       ↓
Train ML Models
       ↓
      ┌───────────────┐
      ↓               ↓
 Clean Test       Shifted Test
      ↓               ↓
   Metrics          Metrics
      └───────┬───────┘
              ↓
      Failure Analysis
              ↓
   Decision Regions
              ↓
       Experiment Report
```

The goal is not simply to determine which model has the highest accuracy.

The goal is to understand **how model behaviour changes when the data distribution changes.**

---

## 📊 Example Investigation

A simple experiment can start with:

```text
Dataset:  Two Moons
Samples:  500
Noise:    0.15
Shift:    0.0
Seed:     42
```

Run the experiment, then increase the shift:

```text
0.0 → 0.5 → 1.5
```

Compare how each model's:

- decision region
- accuracy
- confusion matrix
- error count
- newly introduced failures

change as the distribution moves.

---

## 🛠️ Tech Stack

**Backend**

Python · Flask

**Machine Learning**

scikit-learn · NumPy

**Frontend**

HTML · CSS · JavaScript

**Visualization**

Plotly.js · Chart.js

**Testing**

pytest · unittest

## 🔬 Scientific Scope

ModelShift Lab is an **educational and research-oriented prototype**, not a production robustness benchmark.

The distribution shifts are synthetic and intentionally controlled. They demonstrate specific behaviours but cannot represent every form of real-world data drift.

Results should therefore be interpreted as experimental observations rather than universal claims about model robustness.

The current implementation focuses on:

- Binary classification
- Two-dimensional feature spaces
- Synthetic datasets
- Controlled feature shifts
- Classical machine-learning classifiers

---

## 🚀 Future Direction

Planned extensions include:

- Additional types of distribution shift
- Statistical drift detection
- Multi-class experiments
- Temporal data shift
- Model retraining experiments
- Cross-seed analysis
- Experiment comparison
- Extended robustness reports
- Additional ML algorithms

---

## 💡 Why This Project?

ModelShift Lab was built to explore an important practical ML question:

> **What happens when the data changes after a model has already been trained?**

It combines machine learning, visualization, experimentation, evaluation, and failure analysis into one interactive environment.

---

## 👩‍💻 Author

**Misbah Muhammad Rafique**

Computer Engineering Student | AI & Machine Learning | Software Development

GitHub: **github.com/Misbah81**  
LinkedIn: **linkedin.com/in/misbah-rafique-b6935933a**

---

### ModelShift Lab

**Explore the shift. Measure the failure. Understand the model.**
