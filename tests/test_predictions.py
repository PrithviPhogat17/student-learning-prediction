import unittest

import numpy as np
import pandas as pd

from src.evaluate_model import classification_metrics, evaluate_artifacts
from src.model import ModelBackendError, load_model, tensorflow_available
from src.predict import (ArtifactsMissingError, StudentRiskPredictor, risk_band, validate_input)
from src.schema import get_feature_specs
from src.train_model import run_training
from tests.helpers import make_temp_cfg


def good_input(specs):
    return {s.name: s.default for s in specs}


class ValidationTests(unittest.TestCase):
    specs = get_feature_specs(True)

    def test_valid_input_passes(self):
        clean, errors = validate_input(good_input(self.specs), self.specs)
        self.assertEqual(errors, [])
        self.assertEqual(len(clean), len(self.specs))

    def test_invalid_inputs_give_useful_errors(self):
        bad = good_input(self.specs)
        bad["G1"] = 99
        bad["internet"] = "maybe"
        bad["absences"] = "abc"
        bad["studytime"] = 2.5
        del bad["goout"]
        _, errors = validate_input(bad, self.specs)
        text = " ".join(errors)
        for fragment in ("Period 1 grade", "Internet", "absences", "Weekly study time", "Going out"):
            self.assertIn(fragment.lower(), text.lower())
        self.assertEqual(len(errors), 5)

    def test_nan_rejected(self):
        bad = good_input(self.specs)
        bad["G2"] = float("nan")
        self.assertTrue(validate_input(bad, self.specs)[1])

    def test_risk_band(self):
        self.assertEqual(risk_band(0.1), "lower")
        self.assertEqual(risk_band(0.45), "moderate")
        self.assertEqual(risk_band(0.9), "higher")


class MissingArtifactsTests(unittest.TestCase):
    def test_predictor_without_artifacts_raises_clear_error(self):
        cfg, _ = make_temp_cfg()
        with self.assertRaises(ArtifactsMissingError) as ctx:
            StudentRiskPredictor.from_artifacts(cfg)
        self.assertIn("train.py", str(ctx.exception))

    def test_unknown_backend_rejected(self):
        cfg, _ = make_temp_cfg()
        with self.assertRaises(ValueError):
            run_training(cfg, backend="nonsense", demo=True, verbose=False)


class SklearnFallbackPipelineTests(unittest.TestCase):
    """Full train -> save -> reload -> evaluate -> predict chain with the explicit fallback backend."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.root = make_temp_cfg()
        cls.meta = run_training(cls.cfg, backend="sklearn_mlp", demo=True, verbose=False, seed=7)

    def test_metadata_is_honest_about_backend_and_data(self):
        self.assertEqual(self.meta["backend"], "sklearn_mlp")
        self.assertEqual(self.meta["dataset_mode"], "demo_synthetic")
        self.assertIn("FALLBACK", self.meta["backend_description"])

    def test_history_recorded(self):
        import json
        hist = json.loads((self.root / "reports" / "training_history.json").read_text())
        self.assertGreater(len(hist["loss"]), 1)
        self.assertEqual(len(hist["loss"]), len(hist["val_loss"]))

    def test_reload_gives_identical_predictions(self):
        p1 = StudentRiskPredictor.from_artifacts(self.cfg)
        p2 = StudentRiskPredictor.from_artifacts(self.cfg)
        raw = good_input(p1.specs)
        self.assertAlmostEqual(p1.predict_one(raw)["probability_at_risk"],
                               p2.predict_one(raw)["probability_at_risk"], places=10)

    def test_prediction_output_valid(self):
        p = StudentRiskPredictor.from_artifacts(self.cfg)
        res = p.predict_one(good_input(p.specs))
        self.assertTrue(0.0 <= res["probability_at_risk"] <= 1.0)
        self.assertIn(res["risk_band"], {"lower", "moderate", "higher"})
        test = pd.read_csv(self.root / "processed" / "test.csv")
        probs = p.predict_dataframe(test)
        self.assertEqual(probs.shape, (len(test),))

    def test_better_grade_means_lower_risk(self):
        p = StudentRiskPredictor.from_artifacts(self.cfg)
        low, high = good_input(p.specs), good_input(p.specs)
        low.update(G1=4, G2=3)
        high.update(G1=16, G2=17)
        self.assertGreater(p.predict_one(low)["probability_at_risk"], p.predict_one(high)["probability_at_risk"])

    def test_invalid_input_raises(self):
        p = StudentRiskPredictor.from_artifacts(self.cfg)
        bad = good_input(p.specs)
        bad["G2"] = 50
        with self.assertRaises(ValueError):
            p.predict_one(bad)

    def test_evaluation_runs_and_writes_reports(self):
        report = evaluate_artifacts(self.cfg, n_boot=50, n_repeats=2, verbose=False)
        for name in ("ANN", "Logistic regression", "Majority class"):
            self.assertIn(name, report["test"])
        self.assertTrue((self.root / "reports" / "model_evaluation.json").exists())
        self.assertTrue((self.root / "reports" / "figures" / "confusion_matrix.png").exists())
        self.assertIn("DEMO MODE", " ".join(report["notes"]))

    def test_train_val_test_files_are_disjoint(self):
        ids = {k: set(pd.read_csv(self.root / "processed" / f"{k}.csv")["record_id"]) for k in ("train", "val", "test")}
        self.assertFalse(ids["train"] & ids["test"] or ids["train"] & ids["val"] or ids["val"] & ids["test"])


class MetricTests(unittest.TestCase):
    def test_perfect_and_majority(self):
        y = np.array([0, 0, 1, 1])
        m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["confusion_matrix"], [[2, 0], [0, 2]])
        d = classification_metrics(y, np.zeros(4))
        self.assertEqual(d["recall"], 0.0)


@unittest.skipUnless(tensorflow_available(), "TensorFlow is not installed in this environment")
class KerasTests(unittest.TestCase):
    """Run automatically on a computer where TensorFlow is installed."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.root = make_temp_cfg(max_epochs=15, patience=3)
        cls.meta = run_training(cls.cfg, backend="keras", demo=True, verbose=False, seed=3)

    def test_model_trains_and_reloads(self):
        self.assertEqual(self.meta["backend"], "keras")
        self.assertTrue((self.root / "models" / "ann_model.keras").exists())
        p = StudentRiskPredictor.from_artifacts(self.cfg)
        res = p.predict_one(good_input(p.specs))
        self.assertTrue(0.0 <= res["probability_at_risk"] <= 1.0)

    def test_architecture(self):
        model = load_model("keras", self.root / "models").model
        self.assertEqual(model.outputs[0].shape[-1], 1)
        self.assertEqual(model.inputs[0].shape[-1], self.meta["n_features"])
        self.assertEqual(model.layers[-1].activation.__name__, "sigmoid")

    def test_evaluation_with_keras(self):
        report = evaluate_artifacts(self.cfg, n_boot=30, n_repeats=2, verbose=False)
        self.assertEqual(report["backend"], "keras")


class KerasMissingTests(unittest.TestCase):
    @unittest.skipIf(tensorflow_available(), "TensorFlow is installed")
    def test_clear_error_when_tensorflow_missing(self):
        cfg, _ = make_temp_cfg()
        with self.assertRaises(ModelBackendError) as ctx:
            run_training(cfg, backend="keras", demo=True, verbose=False)
        self.assertIn("sklearn_mlp", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
