
# ModelShift Lab

**Interactive AI Robustness & Failure Analysis Platform** — a local, research-oriented playground for examining how classical machine-learning classifiers respond to synthetic distribution shift.

## Features

- Three reproducible synthetic binary datasets: Two Moons, Concentric Circles, and a configurable class-imbalance dataset.
- Four scikit-learn classifiers: Logistic Regression, RBF SVM, Random Forest, and an MLP neural network.
- Clean versus shifted held-out test evaluation. Models are trained on the original training split; only test features are shifted.
- Actual decision regions computed from each selected trained model over a two-dimensional feature grid.
- Original/shifted sample plot with correct and incorrect shifted predictions.
- Accuracy, precision, recall, F1, and clean/shifted confusion matrices.
- Failure counts: clean errors, shifted errors, newly introduced errors, and samples that become correct after shift.
- Configurable random seed (UI input) for reproducibility.
- Browser-local experiment history (up to 10 runs; click a run to reload its settings) and a full CSV report export (settings, metrics, confusion matrices, failing samples).
- Flask JSON API and automated tests.

## Requirements

- Python 3.10 or newer.
- Internet connection on first page load for Chart.js and Plotly.js CDN assets, unless you download and serve those libraries locally.
- No Node.js, database, cloud account, or hardware required.

## Windows setup (PowerShell)

1. Extract this ZIP to a folder you can edit, e.g. `Documents\ModelShift_Lab`.
2. Open that folder in VS Code.
3. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If PowerShell blocks activation, run the commands using `\.venv\Scripts\python.exe` directly instead.

4. Start the web app:

   ```powershell
   python app.py
   ```

5. Open **http://127.0.0.1:5000** in your browser. Keep the terminal open while using the app. Press `Ctrl+C` to stop it.

If you already use a working Python environment, you can install the requirements there instead of creating `.venv`.

## First experiment

1. Choose **Two Moons**, 500 samples, noise 0.15, shift 0.0, seed 42.
2. Keep all four models selected and click **Run experiment**.
3. Increase shift to 0.5 and rerun; then try 1.5.
4. Try Concentric Circles and Class Imbalance.
5. Compare decision regions, clean and shifted scores, confusion matrices, and the error counts.
6. Click **Export CSV report** to save the current experiment's model metrics and misclassified test samples.
7. Use **Experiment history** to load the settings of a recent run. History is stored in this browser's local storage, not sent to a server.

## Project structure

```text
ModelShift_Lab/
├── app.py                 # Dataset generation, training, evaluation, API
├── requirements.txt       # Python dependencies
├── README.md              # Setup, usage, and interpretation guide
├── LICENSE
├── .gitignore
├── tests/
│   └── test_app.py        # Automated API and reproducibility tests
├── templates/
│   └── index.html         # Main dashboard
└── static/
    ├── styles.css         # Responsive dark UI
    └── app.js              # Charts, history, CSV export
```

## Run tests

From the project root with dependencies installed:

```powershell
python -m pytest -q
```

or:

```powershell
python -m unittest discover -s tests -v
```

The Flask app can also be checked at `http://127.0.0.1:5000/api/health` while it is running.

## API

- `GET /api/health` — confirms the app is running and lists available models/datasets.
- `POST /api/experiment` — accepts JSON fields `dataset`, `samples`, `noise`, `imbalance`, `shift`, `seed`, `models`, and `boundary_model`; returns metrics, confusion matrices, sample coordinates, and decision-region grid data.

## How to interpret results

- **Clean accuracy** measures predictions on held-out test features from the same generated distribution as the training data.
- **Shifted accuracy** measures predictions on the same held-out examples after their features have been deterministically moved.
- **Accuracy change (pp)** is shifted accuracy minus clean accuracy, expressed in percentage points. A negative value means performance decreased.
- Confusion matrices use rows for true labels and columns for predicted labels, ordered as class 0 then class 1.
- A fixed seed makes the generated data/split and stochastic estimators repeatable within the same software environment. Different library versions or platforms may still produce small differences.

## Scientific limitations

This is an educational prototype, not a validated real-world benchmark. The shift function is intentionally synthetic and hand-designed; it does not model every kind of production drift. Synthetic datasets are not evidence of real-world model reliability. Repeat runs across seeds and shift magnitudes, report the setup, and avoid claims that a model is universally robust. The demo is limited to two input features and binary classification. The displayed decision regions are predictions from a trained estimator, not a theoretical boundary.

## Suggested GitHub release checklist

- Run the automated tests in the target environment.
- Add screenshots/GIFs of the dashboard to the repository README after capturing them locally.
- Confirm your Python version and dependency installation instructions.
- Do not commit `.venv/`, `__pycache__/`, `.pytest_cache/`, or personal environment files.
- Add a repository description such as: `Interactive ML robustness experiments with synthetic distribution shift, decision boundaries, failure analysis, and reproducible reports.`

## License

MIT License. See `LICENSE`; replace the placeholder copyright holder with your name before publishing.

## Notes on this version

- Seed is now a UI control (previously hard-coded to 42 in the browser).
- Failure Analysis shows clean and shifted confusion matrices and new/recovered failure counts per model.
- The decision-region grid now covers shifted test points; MLP no longer crashes on tiny minority classes.
- Request validation rejects non-object bodies, non-finite numbers, and non-string model names.
- The API listens on `127.0.0.1:5000` only.
