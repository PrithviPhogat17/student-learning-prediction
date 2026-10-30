import unittest

import numpy as np
import pandas as pd

from src.data_loader import (DatasetNotFoundError, clean_dataframe, inspect_dataframe,
                             load_raw_dataset)
from src.preprocessing import class_weights, fit_transform_splits, split_data
from src.schema import (LABEL_COLUMN, REQUIRED_COLUMNS, TARGET_COLUMN, get_feature_specs)
from tests.helpers import make_temp_cfg


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.root = make_temp_cfg()
        cls.raw = load_raw_dataset(cls.cfg, demo=True)
        cls.df, cls.notes = clean_dataframe(cls.raw, cls.cfg["dataset"]["pass_mark"])

    def test_expected_columns_exist(self):
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, self.raw.columns)

    def test_missing_real_dataset_gives_helpful_error(self):
        with self.assertRaises(DatasetNotFoundError) as ctx:
            load_raw_dataset(self.cfg, demo=False)
        self.assertIn("download_data.py", str(ctx.exception))

    def test_wrong_columns_rejected(self):
        bad = self.root / "demo" / "synthetic_student_mat_DEMO.csv"
        original = bad.read_text()
        try:
            bad.write_text("a,b\n1,2\n")
            with self.assertRaises(ValueError):
                load_raw_dataset(self.cfg, demo=True)
        finally:
            bad.write_text(original)

    def test_target_is_binary_and_matches_rule(self):
        self.assertTrue(set(self.df[LABEL_COLUMN].unique()) <= {0, 1})
        expected = (self.df[TARGET_COLUMN] < self.cfg["dataset"]["pass_mark"]).astype(int)
        self.assertTrue((self.df[LABEL_COLUMN] == expected).all())

    def test_final_grade_is_not_an_input(self):
        for flag in (True, False):
            names = [s.name for s in get_feature_specs(flag)]
            self.assertNotIn(TARGET_COLUMN, names)
            self.assertNotIn(LABEL_COLUMN, names)
        self.assertNotIn("G1", [s.name for s in get_feature_specs(False)])

    def test_inspection_reports_problems(self):
        broken = self.raw.copy()
        broken.loc[0, "G1"] = 99
        broken.loc[1, "internet"] = "maybe"
        report = inspect_dataframe(broken)
        self.assertEqual(report["out_of_range_values"]["G1"], 1)
        self.assertEqual(report["out_of_range_values"]["internet"], 1)

    def test_cleaning_blanks_invalid_values_and_drops_missing_target(self):
        broken = self.raw.copy()
        broken.loc[0, "G1"] = 99
        broken.loc[2, TARGET_COLUMN] = np.nan
        cleaned, notes = clean_dataframe(broken, 10)
        self.assertEqual(notes["rows_dropped_missing_target"], 1)
        self.assertTrue(cleaned["G1"].isna().sum() >= 1)

    def test_splits_do_not_overlap_and_are_stratified(self):
        splits = split_data(self.df, self.cfg)
        ids = [set(splits[k]["record_id"]) for k in ("train", "val", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(sum(len(v) for v in splits.values()), len(self.df))
        overall = self.df[LABEL_COLUMN].mean()
        for part in splits.values():
            self.assertAlmostEqual(part[LABEL_COLUMN].mean(), overall, delta=0.05)

    def test_preprocessor_fitted_on_train_only(self):
        splits = split_data(self.df, self.cfg)
        specs = get_feature_specs(True)
        pre, arrays, labels = fit_transform_splits(splits, specs)
        self.assertEqual(arrays["train"].shape, (len(splits["train"]), len(specs)))
        self.assertEqual(arrays["test"].shape[1], len(specs))
        scaler = pre.named_steps["scale"]
        g2_idx = [s.name for s in specs].index("G2")
        self.assertAlmostEqual(scaler.mean_[g2_idx], splits["train"]["G2"].mean(), places=6)
        self.assertNotAlmostEqual(scaler.mean_[g2_idx], self.df["G2"].mean(), places=6)
        np.testing.assert_allclose(arrays["train"].mean(axis=0), 0, atol=1e-8)

    def test_binary_encoding_and_missing_input_handling(self):
        specs = get_feature_specs(True)
        splits = split_data(self.df, self.cfg)
        pre, _, _ = fit_transform_splits(splits, specs)
        row = splits["test"].iloc[[0]].copy()
        row["internet"] = "unknown-value"      # becomes NaN, then imputed
        out = pre.transform(row)
        self.assertFalse(np.isnan(out).any())
        row2 = splits["test"].iloc[[0]].copy()
        row2["G1"] = np.nan
        self.assertFalse(np.isnan(pre.transform(row2)).any())

    def test_class_weights_balanced(self):
        w = class_weights(np.array([0] * 80 + [1] * 20))
        self.assertAlmostEqual(w[0], 100 / (2 * 80))
        self.assertAlmostEqual(w[1], 100 / (2 * 20))


if __name__ == "__main__":
    unittest.main()
