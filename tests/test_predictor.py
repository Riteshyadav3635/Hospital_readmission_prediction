import unittest

from src.predictor import load_model_bundle, predict
from src.preprocessing import default_inputs, prepare_features
from src.utils import history_frame, make_history_record


class PredictorTests(unittest.TestCase):
    def test_real_artifacts_load_and_match_expected_dimensions(self):
        bundle = load_model_bundle()
        self.assertEqual(len(bundle["feature_columns"]), 41)
        self.assertEqual(bundle["pca"].n_components_, bundle["model"].n_features_in_)

    def test_default_features_produce_a_real_prediction(self):
        bundle = load_model_bundle()
        result = predict(default_inputs(bundle["encoders"]))
        self.assertIn(result["prediction"], (0, 1))
        self.assertIsNotNone(result["probability"])
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)
        self.assertEqual(list(result["feature_frame"].columns), bundle["feature_columns"])

    def test_missing_and_invalid_input_are_rejected(self):
        bundle = load_model_bundle()
        defaults = default_inputs(bundle["encoders"])
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_features({}, bundle["feature_columns"], bundle["encoders"])

        invalid = defaults.copy()
        invalid["time_in_hospital"] = -1
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            prepare_features(invalid, bundle["feature_columns"], bundle["encoders"])

        invalid = defaults.copy()
        invalid["race"] = "not-a-trained-race"
        with self.assertRaisesRegex(ValueError, "Unsupported value for race"):
            prepare_features(invalid, bundle["feature_columns"], bundle["encoders"])

    def test_history_stores_outputs_without_input_features(self):
        record = make_history_record(
            {
                "prediction": 0,
                "probability": 0.25,
                "risk_level": "Lower estimated readmission risk",
            }
        )
        self.assertEqual(
            set(record),
            {"Timestamp (UTC)", "Prediction", "Probability", "Risk level"},
        )
        self.assertEqual(len(history_frame([record])), 1)


if __name__ == "__main__":
    unittest.main()
