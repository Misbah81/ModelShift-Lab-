# ModelShift Lab

### Interactive ML Robustness & Failure Analysis Platform

**ModelShift Lab** is a local, research-oriented machine-learning laboratory for studying how classical classifiers behave when the distribution of test data changes.

Instead of looking at accuracy on a single dataset, ModelShift Lab lets you run controlled experiments, visualize learned decision regions, compare clean and shifted test performance, inspect classification failures, and export reproducible experiment results.

> **Core idea:** A model performing well on clean test data does not automatically mean it will perform well when the input distribution changes.

---

## ✨ Why ModelShift Lab?

Machine-learning models are usually evaluated on data that is assumed to resemble the training distribution.

But what happens when that assumption no longer holds?

ModelShift Lab creates controlled synthetic experiments where the model is:

**trained once → evaluated on clean data → evaluated again after a controlled feature shift → analyzed for failures**

This makes model behavior easier to observe and investigate.

---

## 🔬 What You Can Explore

### 1. Synthetic Datasets

Experiment with three binary classification problems:

- **Two Moons** — nonlinear, curved class boundaries
- **Concentric Circles** — radial nonlinear separation
- **Class Imbalance** — configurable minority/majority class distribution

Dataset parameters can be changed directly from the interface.

---

### 2. Multiple ML Models

Compare four classical scikit-learn classifiers:

| Model | Learning Behaviour |
|---|---|
| Logistic Regression | Linear decision boundary |
| RBF SVM | Nonlinear kernel-based separation |
| Random Forest | Ensemble of decision trees |
| MLP Neural Network | Learned nonlinear representation |

The same experiment configuration can be applied to multiple models, making comparisons easier.

---

### 3. Clean vs Shifted Evaluation

Models are trained using the original training distribution.

The held-out test set is then evaluated in two conditions:

**Clean test data**

The original test features.

**Shifted test data**

The same held-out samples after a deterministic synthetic feature shift.

This allows the experiment to measure how much performance changes under the selected shift.

---

### 4. Decision Regions

ModelShift Lab generates an actual two-dimensional prediction grid using the selected trained estimator.

The resulting visualization shows:

- learned decision regions
- test samples
- class labels
- correct predictions
- incorrect predictions
- shifted samples

The displayed boundary is therefore based on the model's predictions rather than a manually drawn illustration.

---

### 5. Failure Analysis

The platform does more than display accuracy.

For each model, you can inspect:

- Clean accuracy
- Shifted accuracy
- Precision
- Recall
- F1 score
- Clean confusion matrix
- Shifted confusion matrix
- Clean errors
- Shifted errors
- Newly introduced errors
- Samples that become correct after shifting

This helps answer a more useful question:

> **What actually changed when the distribution shifted?**

---

### 6. Reproducible Experiments

Each experiment has a configurable random seed.

For example:

```text
Dataset:       Two Moons
Samples:       500
Noise:         0.15
Shift:         0.50
Seed:          42
Models:        Logistic Regression, SVM, Random Forest, MLP
```

Using the same configuration and seed makes the experiment reproducible within the same software environment.

---

### 7. Experiment History

The browser stores up to **10 recent experiments**.

You can:

- review previous configurations
- reload an experiment
- compare different shift settings
- reproduce previous runs

History is stored locally in the browser and is not uploaded to a server.

---

### 8. CSV Experiment Reports

Export the current experiment as a CSV report containing:

- experiment settings
- model metrics
- confusion matrices
- failure counts
- misclassified test samples

This makes it easier to continue analysis outside the application.

---

## 🧪 Example Experiment

Try the following experiment:

### Experiment A — Baseline

```text
Dataset: Two Moons
Samples: 500
Noise: 0.15
Shift: 0.0
Seed: 42
```

Run all four models.

Record their clean performance.

### Experiment B — Moderate Shift

Change:

```text
Shift: 0.5
```

Run the experiment again.

Compare:

- clean accuracy
- shifted accuracy
- confusion matrices
- decision regions
- failure counts

### Experiment C — Stronger Shift

Try:

```text
Shift: 1.5
```

Now investigate whether the performance degradation becomes larger.

The goal is not to find a model that "wins."

The goal is to **observe and explain model behaviour under controlled distribution changes.**

---

## 📊 What the Results Mean

### Clean Accuracy

Performance on held-out test features from the original generated distribution.

### Shifted Accuracy

Performance on the same held-out examples after their features have been deterministically shifted.

### Accuracy Change

```text
Shifted Accuracy − Clean Accuracy
```

A negative value indicates that performance decreased after the selected shift.

### Confusion Matrix

Rows represent the true class.

Columns represent the predicted class.

Classes are ordered:

```text
0
1
```

### Failure Counts

The application distinguishes between:

- errors already present on clean data
- errors after shifting
- newly introduced errors
- predictions that become correct after shifting

This gives more information than accuracy alone.

---

## 🧠 Technical Architecture

```text
                    ┌─────────────────────┐
                    │    Browser UI       │
                    │ HTML / CSS / JS     │
                    └──────────┬──────────┘
                               │
                         JSON API
                               │
                    ┌──────────▼──────────┐
                    │      Flask          │
                    │  Experiment API     │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       Dataset Engine     Model Training    Evaluation
             │                 │                 │
             ▼                 ▼                 ▼
       Synthetic Data    scikit-learn      Clean / Shifted
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Failure Analysis    │
                    │ Decision Regions    │
                    │ Metrics / Reports   │
                    └─────────────────────┘
```

---

## 🛠️ Technology Stack

**Frontend**

- HTML
- CSS
- JavaScript
- Chart.js
- Plotly.js

**Backend**

- Python
- Flask

**Machine Learning**

- scikit-learn
- NumPy
- Pandas

**Testing**

- pytest
- unittest

**Storage**

- Browser Local Storage for experiment history

No external database is required.

---

## 📁 Project Structure

```text
ModelShift_Lab/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── tests/
│   └── test_app.py
│
├── templates/
│   └── index.html
│
└── static/
    ├── styles.css
    └── app.js
```

### Main Files

| File | Purpose |
|---|---|
| `app.py` | Flask application, datasets, models, experiments and API |
| `index.html` | Main dashboard interface |
| `styles.css` | Dashboard styling and responsive layout |
| `app.js` | Experiment controls, charts, history and CSV export |
| `test_app.py` | Automated API and reproducibility tests |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Installation

### Requirements

- Python **3.10+**
- Modern web browser
- Internet connection on first page load for CDN-hosted Chart.js and Plotly.js assets

Node.js is **not required**.

No:

- IoT hardware
- database
- cloud account
- external ML service

is required.

---

## 🪟 Windows Setup

Clone or download the repository.

Open the project folder in VS Code.

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the application:

```powershell
python app.py
```

Open:

**http://127.0.0.1:5000/**

Keep the terminal running while using the application.

Stop the server with:

```text
Ctrl + C
```

### If PowerShell blocks virtual-environment activation

You can directly use the virtual environment's Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

and:

```powershell
.\.venv\Scripts\python.exe app.py
```

---

## 🧪 Run Tests

Using pytest:

```powershell
python -m pytest -q
```

Or using unittest:

```powershell
python -m unittest discover -s tests -v
```

---

## 🔌 API

### Health Check

```http
GET /api/health
```

Returns application status and available datasets/models.

While the Flask application is running:

```text
http://127.0.0.1:5000/api/health
```

### Experiment Endpoint

```http
POST /api/experiment
```

Supported input fields include:

```text
dataset
samples
noise
imbalance
shift
seed
models
boundary_model
```

The endpoint returns experiment metrics, confusion matrices, sample coordinates, predictions and decision-region grid data.

---

## 🔍 Scientific Interpretation

ModelShift Lab is intentionally designed as an **educational and experimental platform**, not as a production robustness benchmark.

The shift mechanism is synthetic and hand-designed.

Therefore:

> A model performing well in these experiments does not mean that it is universally robust to real-world distribution shift.

Likewise, poor performance under one synthetic shift does not automatically mean that the model will fail in every real-world scenario.

Meaningful experimentation should involve:

- multiple random seeds
- multiple shift magnitudes
- multiple datasets
- multiple models
- clearly documented configurations
- appropriate evaluation metrics

The objective is to **measure behaviour rather than make unsupported robustness claims.**

---

## ⚠️ Current Limitations

The current version focuses on:

- two input features
- binary classification
- synthetic datasets
- synthetic feature shifts
- scikit-learn classifiers

It does not attempt to model every form of real-world distribution shift.

The decision-region visualization represents predictions from a trained estimator over a two-dimensional grid; it is not a theoretical representation of the model's mathematical boundary.

Different Python, scikit-learn, NumPy or platform versions may produce small numerical differences even when the same seed is used.

---

## 🚀 Future Development

Potential future directions include:

- Additional drift types
- More statistical drift detectors
- Multi-class experiments
- Temporal distribution shift
- Model retraining experiments
- Cross-seed statistical summaries
- Experiment comparison views
- Automated experiment reports
- More classifiers
- Larger benchmark datasets
- Interactive failure inspection
- Robustness curves across shift magnitude

---

## 🎯 Learning Goals

This project was developed to explore practical concepts in:

- Machine Learning
- Model Evaluation
- Distribution Shift
- Decision Boundaries
- Classification Failure Analysis
- Reproducible Experiments
- Data Visualization
- Flask API Development
- Scientific Experiment Design

The project is intentionally built around **experimentation and observation**, rather than treating a single accuracy score as the complete story.

---

## 📌 Project Status

**Status:** Active Learning / Research Prototype

The project is being developed incrementally, with each stage adding more meaningful experimentation and analysis.

---

## 📜 License

MIT License.

See `LICENSE` for details.

---

## 👩‍💻 Author

**Misbah Muhammad Rafique**

Computer Engineering Student  
Interested in Machine Learning, AI, Software Development and Intelligent Systems.

GitHub: `https://github.com/Misbah81`

LinkedIn: `https://linkedin.com/in/misbah-rafique-b6935933a`

---

> **ModelShift Lab — Don't just measure accuracy. Investigate what changes when the data changes.**
