"""Create a clearly labelled SYNTHETIC demo CSV (not real students) in data/demo/."""
import _bootstrap  # noqa: F401

from src.demo_data import DEMO_FILENAME, generate_demo_dataframe
from src.utils import load_config, resolve_path

if __name__ == "__main__":
    cfg = load_config()
    out_dir = resolve_path(cfg["dataset"]["demo_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / DEMO_FILENAME
    generate_demo_dataframe().to_csv(path, index=False)
    print(f"Wrote synthetic DEMO data to {path}\nThis is NOT real student data; results from it are illustrative only.")
