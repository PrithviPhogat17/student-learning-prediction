# Windows PowerShell helper: creates a virtual environment and installs dependencies.
# Usage:  .\setup_windows.ps1
$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "`nSetup complete. Next:  python scripts/run_pipeline.py   then   streamlit run app.py"
