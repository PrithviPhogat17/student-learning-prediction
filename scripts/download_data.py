"""Download the UCI Student Performance dataset (CC BY 4.0) into data/raw/.

Source: https://archive.ics.uci.edu/dataset/320/student+performance
Cite as: Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository.
         https://doi.org/10.24432/C5TG7T

NOTE: this script needs internet access. It could not be executed in the environment
where the project was written, so if it fails use the manual steps printed at the end.
"""
import _bootstrap  # noqa: F401  (sets sys.path)

import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.utils import load_config, resolve_path

URLS = [
    "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
]
WANTED = ("student-mat.csv", "student-por.csv")


def extract_all(data: bytes, dest: Path) -> None:
    """Extract a zip, including zips nested inside it (the UCI archive contains student.zip)."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            payload = zf.read(name)
            if name.lower().endswith(".zip"):
                extract_all(payload, dest)
            elif Path(name).name in WANTED:
                (dest / Path(name).name).write_bytes(payload)


def main() -> int:
    cfg = load_config()
    dest = resolve_path(cfg["dataset"]["raw_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    needed = dest / cfg["dataset"]["course_file"]
    if needed.exists():
        print(f"Already present: {needed}")
        return 0
    for url in URLS:
        try:
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                extract_all(resp.read(), dest)
            if needed.exists():
                print(f"Saved {needed}")
                return 0
        except Exception as exc:  # network errors, bad zip, ...
            print(f"  failed: {exc}")
    print("\nAutomatic download failed. Manual steps:\n"
          "  1. Open https://archive.ics.uci.edu/dataset/320/student+performance and click Download.\n"
          "  2. Unzip it (it contains a nested student.zip - unzip that too).\n"
          f"  3. Copy {cfg['dataset']['course_file']} into {dest}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
