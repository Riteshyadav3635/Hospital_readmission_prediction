import unittest

from streamlit.testing.v1 import AppTest

from src.utils import HISTORY_PATH, clear_prediction_history, load_prediction_history


PAGES = [
    "🏠 Dashboard",
    "🩺 Predict Readmission",
    "📊 Analytics",
    "📋 Prediction History",
    "🧠 Model Insights",
    "📂 Dataset Explorer",
    "ℹ️ About Project",
]


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        clear_prediction_history()

    def tearDown(self):
        clear_prediction_history()

    def test_all_major_pages_render_without_streamlit_exceptions(self):
        app = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Hospital Readmission Prediction" in item.value for item in app.title))
        self.assertIn("Start Prediction →", [button.label for button in app.button])
        next(button for button in app.button if button.label == "Start Prediction →").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value, "🩺 Predict Readmission")

        for page in PAGES[1:]:
            with self.subTest(page=page):
                app.radio[0].set_value(page).run()
                self.assertFalse(app.exception)
                if page in {"📊 Analytics", "📂 Dataset Explorer"}:
                    self.assertEqual(len(app.get("download_button")), 1)

        app.radio[0].set_value("📊 Analytics").run()
        baseline_caption = next(c.value for c in app.caption if c.value.startswith("Showing "))
        age_filter = next(widget for widget in app.selectbox if widget.key == "shared_age_group")
        age_filter.set_value("[0-10)").run()
        filtered_caption = next(c.value for c in app.caption if c.value.startswith("Showing "))
        self.assertNotEqual(filtered_caption, baseline_caption)
        app.radio[0].set_value("🏠 Dashboard").run()
        dashboard_caption = next(c.value for c in app.caption if c.value.startswith("Showing "))
        self.assertEqual(dashboard_caption, filtered_caption)
        app.radio[0].set_value("📂 Dataset Explorer").run()
        explorer_caption = next(c.value for c in app.caption if c.value.startswith("Showing "))
        self.assertEqual(explorer_caption, filtered_caption)

    def test_real_prediction_is_saved_without_input_values_and_can_be_cleared(self):
        app = AppTest.from_file("app.py", default_timeout=60).run()
        app.radio[0].set_value("🩺 Predict Readmission").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.number_input), 11)
        self.assertIn("Generate Prediction", [button.label for button in app.button])

        next(button for button in app.button if button.label == "Generate Prediction").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(metric.label == "Prediction" for metric in app.metric))
        self.assertEqual(len(app.get("download_button")), 1)

        stored = load_prediction_history()
        self.assertEqual(len(stored), 1)
        self.assertEqual(
            list(stored.columns),
            ["Timestamp (UTC)", "Prediction", "Probability", "Risk level"],
        )
        self.assertFalse(any("patient" in column.lower() for column in stored.columns))

        next(button for button in app.button if button.label == "View Prediction History →").click().run()
        self.assertEqual(app.radio[0].value, "📋 Prediction History")
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.get("download_button")), 1)
        next(button for button in app.button if button.label == "Clear prediction history").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(load_prediction_history().empty)
        self.assertFalse(HISTORY_PATH.exists())
        next(button for button in app.button if button.label == "← Back to Dashboard").click().run()
        self.assertEqual(app.radio[0].value, "🏠 Dashboard")
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
