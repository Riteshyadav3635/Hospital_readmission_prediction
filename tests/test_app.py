import unittest

from streamlit.testing.v1 import AppTest

DISCLAIMER = (
    "This application is intended for educational and research purposes only and "
    "should not be used as a substitute for professional medical advice."
)


class StreamlitAppTests(unittest.TestCase):
    def test_home_renders_and_loads_the_saved_model(self):
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("Hospital Readmission Prediction", [item.value for item in app.title])
        self.assertTrue(any(DISCLAIMER in item.value for item in app.warning))

    def test_prediction_and_session_history_can_be_cleared(self):
        app = AppTest.from_file("app.py", default_timeout=30).run()
        app.radio[0].set_value("Prediction").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.number_input), 11)
        self.assertEqual(app.button[0].label, "Run readmission estimate")

        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        self.assertEqual(len(app.metric), 2)
        self.assertEqual(len(app.session_state["history"]), 1)

        app.radio[0].set_value("History").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["history"]), 0)
        self.assertTrue(any("No predictions" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
