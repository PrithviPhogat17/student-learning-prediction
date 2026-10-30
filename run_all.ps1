# One-command setup and run for Windows PowerShell (NOT tested by the author - no Windows/internet available).
# Usage:  .\run_all.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_setup.py
python scripts/run_pipeline.py                      # download, EDA, Keras training, evaluation
python -m unittest discover -s tests -t . -v        # all tests incl. Keras and Streamlit smoke tests
streamlit run app.py
