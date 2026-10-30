"""Train the ANN and save the model, preprocessor and metadata.

Examples (PowerShell):
    python scripts/train.py                        # real data, TensorFlow/Keras ANN
    python scripts/train.py --no-prior-grades      # harder task without G1/G2
    python scripts/train.py --backend sklearn_mlp  # explicit fallback if TensorFlow cannot be installed
    python scripts/train.py --demo                 # synthetic demo data (illustrative only)
"""
import _bootstrap  # noqa: F401

import argparse
import sys

from src.data_loader import DatasetNotFoundError
from src.model import BACKENDS, ModelBackendError
from src.train_model import run_training
from src.utils import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=BACKENDS, default="keras")
    ap.add_argument("--demo", action="store_true", help="use the synthetic demo dataset")
    ap.add_argument("--no-prior-grades", action="store_true", help="exclude G1 and G2 from the inputs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    try:
        run_training(load_config(args.config), backend=args.backend, demo=args.demo,
                     include_prior_grades=False if args.no_prior_grades else None, seed=args.seed)
    except (DatasetNotFoundError, ModelBackendError) as exc:
        print(f"\nERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
