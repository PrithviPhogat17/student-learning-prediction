"""Streamlit smoke test. Skipped automatically when Streamlit is not installed.

NOTE: this file was written in an environment WITHOUT Streamlit, so it has not been executed
by the author. The page logic itself was exercised with a stub (see README, 'What was tested').
"""
import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except Exception:  # Streamlit missing or too old
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app.py", "pages/1_Performance_Analysis.py", "pages/2_Prediction.py",
         "pages/3_Recommendations.py", "pages/4_Model_Insights.py", "pages/5_About.py"]


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class PageSmokeTests(unittest.TestCase):
    def test_pages_run_without_exceptions(self):
        for page in PAGES:
            with self.subTest(page=page):
                at = AppTest.from_file(str(ROOT / page), default_timeout=60).run()
                self.assertFalse(at.exception, f"{page}: {at.exception}")


if __name__ == "__main__":
    unittest.main()
