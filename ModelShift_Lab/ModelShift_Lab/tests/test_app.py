import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, create_dataset, shift_features  # noqa: E402


class ModelShiftTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def post(self, payload):
        return self.client.post("/api/experiment", json=payload)

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ModelShift Lab", response.data)

    def test_health_endpoint(self):
        data = self.client.get("/api/health").get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("SVM (RBF)", data["models"])

    def test_invalid_dataset_rejected(self):
        self.assertEqual(self.post({"dataset": "nope"}).status_code, 400)

    def test_no_model_rejected(self):
        self.assertEqual(self.post({"models": []}).status_code, 400)

    def test_unhashable_models_rejected(self):
        self.assertEqual(self.post({"models": [["x"], {"a": 1}]}).status_code, 400)

    def test_non_object_body_rejected(self):
        self.assertEqual(self.post([1, 2, 3]).status_code, 400)

    def test_bad_numbers_rejected(self):
        for bad in ("abc", None, True, float("inf")):
            r = self.post({"models": ["Logistic Regression"], "noise": bad})
            self.assertEqual(r.status_code, 400, bad)
        r = self.client.post("/api/experiment", data='{"noise": NaN}', content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_dataset_reproducibility(self):
        x1, y1 = create_dataset("moons", 150, 0.1, 0.5, 7)
        x2, y2 = create_dataset("moons", 150, 0.1, 0.5, 7)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_zero_shift_keeps_features(self):
        X = np.array([[0.0, 1.0], [2.0, 3.0]])
        np.testing.assert_array_equal(shift_features(X, 0), X)

    def test_shift_changes_features_without_mutating_input(self):
        X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        original = X.copy()
        shifted = shift_features(X, 1.0)
        np.testing.assert_array_equal(X, original)
        self.assertFalse(np.array_equal(shifted, X))

    def test_single_model_experiment_schema(self):
        response = self.post({
            "dataset": "moons", "samples": 120, "noise": 0.1, "shift": 0.3,
            "seed": 11, "models": ["Logistic Regression"], "boundary_model": "Logistic Regression",
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        data = response.get_json()
        self.assertEqual(data["summary"]["seed"], 11)
        self.assertEqual(data["boundary"]["model"], "Logistic Regression")
        self.assertEqual(len(data["selected_analysis"]["shift_confusion"]), 2)
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(len(data["test_points"]), data["summary"]["test_samples"])

    def test_experiment_is_reproducible(self):
        payload = {"dataset": "circles", "samples": 200, "shift": 0.5, "seed": 3,
                   "models": ["Random Forest", "MLP Neural Network"]}
        a, b = self.post(payload).get_json(), self.post(payload).get_json()
        self.assertEqual(a["results"], b["results"])

    def test_extreme_imbalance_small_sample_all_models(self):
        r = self.post({"dataset": "classification", "samples": 100, "imbalance": 0.05, "seed": 1})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))

    def test_boundary_grid_covers_shifted_points(self):
        data = self.post({"dataset": "moons", "samples": 200, "shift": 2.0,
                          "models": ["Logistic Regression"]}).get_json()
        xs = [p["shift_x"] for p in data["test_points"]]
        ys = [p["shift_y"] for p in data["test_points"]]
        b = data["boundary"]
        self.assertLessEqual(max(xs), b["x"][-1])
        self.assertGreaterEqual(min(ys), b["y"][0])

    def test_failure_counts_consistent(self):
        data = self.post({"dataset": "moons", "samples": 300, "shift": 1.0,
                          "models": ["SVM (RBF)"]}).get_json()
        r = data["results"][0]
        self.assertEqual(r["shift_errors"], r["clean_errors"] + r["new_failures_after_shift"] - r["recovered_after_shift"])


if __name__ == "__main__":
    unittest.main()
