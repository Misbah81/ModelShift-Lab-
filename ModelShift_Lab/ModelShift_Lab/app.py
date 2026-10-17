"""ModelShift Lab: reproducible synthetic distribution-shift experiments."""
from __future__ import annotations

import math

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)

MODELS = {
    "Logistic Regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "SVM (RBF)": lambda seed: SVC(kernel="rbf", random_state=seed),
    "Random Forest": lambda seed: RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=1),
    # early_stopping is off: its internal stratified validation split crashes on tiny minority classes.
    "MLP Neural Network": lambda seed: MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=700, random_state=seed
    ),
}
DATASETS = {"moons", "circles", "classification"}


def create_dataset(kind: str, samples: int, noise: float, imbalance: float, seed: int):
    if kind == "moons":
        return make_moons(n_samples=samples, noise=noise, random_state=seed)
    if kind == "circles":
        return make_circles(n_samples=samples, noise=noise, factor=0.45, random_state=seed)
    if kind == "classification":
        p = float(np.clip(imbalance, 0.05, 0.95))
        return make_classification(
            n_samples=samples, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
            n_classes=2, n_clusters_per_class=1, class_sep=1.0,
            flip_y=min(noise * 0.35, 0.3), weights=[p, 1 - p], random_state=seed,
        )
    raise ValueError("Unknown dataset. Choose moons, circles, or classification.")


def shift_features(X: np.ndarray, amount: float) -> np.ndarray:
    """Deterministic covariate shift applied to evaluation features only.

    Points right of the test-set median on feature 1 move right by `amount`;
    the remaining points move down on feature 2 by 0.35 * `amount`.
    """
    shifted = X.copy()
    if amount > 0:
        mask = shifted[:, 0] > np.median(shifted[:, 0])
        shifted[mask, 0] += amount
        shifted[~mask, 1] -= amount * 0.35
    return shifted


def model_metrics(name, model, X, y):
    pred = model.predict(X)
    return {
        "name": name,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }, pred


def _number(value, cast, low, high):
    """Parse a finite number and clamp it to [low, high]."""
    if isinstance(value, bool):
        raise ValueError("booleans are not numbers")
    number = cast(value)
    if isinstance(number, float) and not math.isfinite(number):
        raise ValueError("number must be finite")
    return max(low, min(high, number))


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "models": list(MODELS), "datasets": sorted(DATASETS)})


@app.post("/api/experiment")
def experiment():
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    kind = data.get("dataset", "moons")
    if not isinstance(kind, str) or kind not in DATASETS:
        return jsonify({"error": "Invalid dataset selection."}), 400
    try:
        samples = _number(data.get("samples", 500), int, 100, 1500)
        noise = _number(data.get("noise", 0.15), float, 0.0, 0.5)
        imbalance = _number(data.get("imbalance", 0.5), float, 0.05, 0.95)
        seed = _number(data.get("seed", 42), int, 0, 100000)
        shift = _number(data.get("shift", 0.0), float, 0.0, 2.0)
    except (TypeError, ValueError, OverflowError):
        return jsonify({"error": "Experiment parameters must be valid numbers."}), 400

    requested = data.get("models", list(MODELS))
    if not isinstance(requested, list):
        return jsonify({"error": "Models must be provided as a list."}), 400
    names = list(dict.fromkeys(n for n in requested if isinstance(n, str) and n in MODELS))
    if not names:
        return jsonify({"error": "Select at least one valid model."}), 400
    boundary_name = data.get("boundary_model", names[0])
    if not isinstance(boundary_name, str) or boundary_name not in names:
        boundary_name = names[0]

    X, y = create_dataset(kind, samples, noise, imbalance, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    X_shifted = shift_features(X_test, shift)

    fitted, results, predictions = {}, [], {}
    for name in names:
        model = make_pipeline(StandardScaler(), MODELS[name](seed))
        model.fit(X_train, y_train)
        fitted[name] = model
        clean, clean_pred = model_metrics(name, model, X_test, y_test)
        shifted, shifted_pred = model_metrics(name, model, X_shifted, y_test)
        results.append({
            "name": name,
            "clean_accuracy": clean["accuracy"],
            "accuracy": shifted["accuracy"],
            "clean_precision": clean["precision"],
            "clean_recall": clean["recall"],
            "clean_f1": clean["f1"],
            "precision": shifted["precision"],
            "recall": shifted["recall"],
            "f1": shifted["f1"],
            "clean_confusion": clean["confusion"],
            "confusion": shifted["confusion"],
            "accuracy_change_pp": round((shifted["accuracy"] - clean["accuracy"]) * 100, 2),
            "clean_errors": int(np.sum(clean_pred != y_test)),
            "shift_errors": int(np.sum(shifted_pred != y_test)),
            "new_failures_after_shift": int(np.sum((clean_pred == y_test) & (shifted_pred != y_test))),
            "recovered_after_shift": int(np.sum((clean_pred != y_test) & (shifted_pred == y_test))),
        })
        predictions[name] = {"clean": clean_pred, "shifted": shifted_pred}

    model = fitted[boundary_name]
    # Grid covers both original and shifted test points (shift moves x up to +shift, y down to -0.35*shift).
    x_min = float(min(X[:, 0].min(), X_shifted[:, 0].min()) - 0.8)
    x_max = float(max(X[:, 0].max(), X_shifted[:, 0].max()) + 0.8)
    y_min = float(min(X[:, 1].min(), X_shifted[:, 1].min()) - 0.8)
    y_max = float(max(X[:, 1].max(), X_shifted[:, 1].max()) + 0.8)
    gx, gy = np.linspace(x_min, x_max, 65), np.linspace(y_min, y_max, 65)
    xx, yy = np.meshgrid(gx, gy)
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int).reshape(len(gy), len(gx))

    clean_pred = predictions[boundary_name]["clean"]
    shifted_pred = predictions[boundary_name]["shifted"]
    train_points = [
        {"x": round(float(X_train[i, 0]), 4), "y": round(float(X_train[i, 1]), 4), "label": int(y_train[i])}
        for i in range(len(X_train))
    ]
    test_points = [{
        "x": round(float(X_test[i, 0]), 4), "y": round(float(X_test[i, 1]), 4),
        "shift_x": round(float(X_shifted[i, 0]), 4), "shift_y": round(float(X_shifted[i, 1]), 4),
        "label": int(y_test[i]), "clean_pred": int(clean_pred[i]), "shift_pred": int(shifted_pred[i]),
        "clean_correct": bool(clean_pred[i] == y_test[i]), "shift_correct": bool(shifted_pred[i] == y_test[i]),
    } for i in range(len(X_test))]
    failed = [i for i, p in enumerate(test_points) if not p["shift_correct"]]

    return jsonify({
        "results": results,
        "train_points": train_points,
        "test_points": test_points,
        "boundary": {"model": boundary_name, "x": gx.round(4).tolist(), "y": gy.round(4).tolist(), "z": zz.tolist()},
        "selected_analysis": {
            "model": boundary_name,
            "clean_confusion": confusion_matrix(y_test, clean_pred, labels=[0, 1]).tolist(),
            "shift_confusion": confusion_matrix(y_test, shifted_pred, labels=[0, 1]).tolist(),
            "misclassified_count": len(failed),
            "misclassified_indices": failed,
            "clean_misclassified_count": int(np.sum(clean_pred != y_test)),
            "new_failures_after_shift": int(np.sum((clean_pred == y_test) & (shifted_pred != y_test))),
            "recovered_after_shift": int(np.sum((clean_pred != y_test) & (shifted_pred == y_test))),
        },
        "summary": {
            "dataset": kind, "samples": samples, "noise": noise, "imbalance": imbalance,
            "shift": shift, "seed": seed, "test_samples": len(y_test),
            "train_samples": len(y_train), "models": names, "boundary_model": boundary_name,
        },
    })


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "An unexpected server error occurred. Check the terminal for details."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
