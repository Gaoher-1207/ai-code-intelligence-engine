"""Streamlit app smoke tests using the normal project imports and real models."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "frontend" / "app.py"


class FrontendIntegrationTests(unittest.TestCase):
    def _run_app(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        return app

    def test_python_ui_runs_gaoher_and_hajera_prediction_pipeline(self) -> None:
        app = self._run_app()
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertGreaterEqual(len(app.metric), 2)
        self.assertEqual(app.metric[0].label, "Estimated time complexity")
        self.assertIn(app.metric[0].value, {"O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "exponential / NP-hard"})

    def test_c_ui_runs_hajera_defect_prediction_pipeline(self) -> None:
        app = self._run_app()
        app.radio[0].set_value("C").run()
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertEqual(len(app.metric), 1)
        self.assertEqual(app.metric[0].label, "Defect risk")
        self.assertTrue(app.metric[0].value.endswith("%"))
        self.assertIn(app.metric[0].delta, {"vulnerable", "non-vulnerable"})


if __name__ == "__main__":
    unittest.main()
