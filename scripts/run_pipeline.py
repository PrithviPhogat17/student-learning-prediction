"""One command for the whole workflow: download -> EDA -> train -> evaluate.

    python scripts/run_pipeline.py                   # real data + Keras ANN
    python scripts/run_pipeline.py --backend sklearn_mlp
    python scripts/run_pipeline.py --demo            # synthetic data (illustrative only)
"""
import _bootstrap  # noqa: F401

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(HERE / script), *extra]
    print("\n>>", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"Step failed: {script}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="keras", choices=["keras", "sklearn_mlp"])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-prior-grades", action="store_true")
    a = ap.parse_args()
    demo = ["--demo"] if a.demo else []
    if a.demo:
        run("make_demo_data.py")
    else:
        run("download_data.py")
    run("run_eda.py", *demo)
    train_args = ["--backend", a.backend, *demo] + (["--no-prior-grades"] if a.no_prior_grades else [])
    run("train.py", *train_args)
    run("evaluate.py")
    print("\nAll done. Start the dashboard with:  streamlit run app.py")
