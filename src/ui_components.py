"""Streamlit-only helpers (layout, styling, cached loaders). Logic lives in the other modules."""
from __future__ import annotations

from types import SimpleNamespace

import streamlit as st

from .data_loader import resolve_dataset_for_app
from .predict import StudentRiskPredictor
from .train_model import METADATA_FILE
from .utils import get_dirs, load_config, read_json

NAVY, TEAL = "#0B2545", "#13A89E"

CSS = f"""
<style>
h1, h2, h3 {{ color: {NAVY}; }}
.kpi {{ background: white; border-left: 5px solid {TEAL}; border-radius: 8px; padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(11,37,69,.12); margin-bottom: 8px; }}
.kpi .label {{ font-size: .80rem; color: #5B6B7F; text-transform: uppercase; letter-spacing: .04em; }}
.kpi .value {{ font-size: 1.7rem; font-weight: 700; color: {NAVY}; }}
.kpi .note {{ font-size: .78rem; color: #5B6B7F; }}
.hero {{ background: linear-gradient(90deg, {NAVY}, #14587A); color: white; padding: 22px 26px;
         border-radius: 10px; margin-bottom: 14px; }}
.hero h1 {{ color: white; margin: 0; font-size: 1.9rem; }}
.hero p {{ margin: 4px 0 0 0; color: #CFE8E6; }}
.badge {{ display:inline-block; padding: 2px 10px; border-radius: 12px; color: white; font-size: .78rem; }}
</style>
"""


def setup_page(title: str, icon: str) -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} | Student Learning Analytics", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def kpi_card(label: str, value: str, note: str = "") -> None:
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="note">{note}</div></div>', unsafe_allow_html=True)


def priority_badge(priority: str) -> str:
    color = {"high": "#C0392B", "medium": "#D68910", "low": TEAL}[priority]
    return f'<span class="badge" style="background:{color}">{priority} priority</span>'


def demo_banner(metadata: dict | None) -> None:
    if metadata and metadata.get("dataset_mode") == "demo_synthetic":
        st.warning("DEMO MODE: this model was trained on SYNTHETIC data. Results are illustrative only "
                   "and say nothing about real students.")
    if metadata and metadata.get("backend") != "keras":
        st.info(f"Model backend: {metadata.get('backend_description')}. The main project model is the "
                "TensorFlow/Keras ANN - train it with `python scripts/train.py`.")


def empty_state(message: str) -> None:
    st.info(message)


@st.cache_resource(show_spinner="Loading trained model...")
def _load_predictor(_stamp: float):
    try:
        return StudentRiskPredictor.from_artifacts(), None
    except Exception as exc:  # missing files, TensorFlow import problems, corrupt files
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _load_dataset(_stamp: float, _cfg: dict, _metadata: dict | None):
    return resolve_dataset_for_app(_cfg, _metadata)


def get_context() -> SimpleNamespace:
    """Config, folders, predictor (or the reason there is none), metadata."""
    cfg = load_config()
    dirs = get_dirs(cfg)
    meta_path = dirs["models"] / METADATA_FILE
    stamp = meta_path.stat().st_mtime if meta_path.exists() else 0.0
    predictor, error = _load_predictor(stamp)
    metadata = predictor.metadata if predictor else (read_json(meta_path) if meta_path.exists() else None)
    if predictor is None and error is None:
        error = "No trained model found. Run: python scripts/train.py"
    df, mode = _load_dataset(stamp, cfg, metadata)
    return SimpleNamespace(cfg=cfg, dirs=dirs, predictor=predictor, error=error,
                           metadata=metadata, data=df, data_mode=mode)


def load_report(ctx) -> dict | None:
    path = ctx.dirs["reports"] / "model_evaluation.json"
    return read_json(path) if path.exists() else None


NO_MODEL_HELP = ("The trained model is not available yet.\n\n"
                 "1. `python scripts/download_data.py`  (or `python scripts/make_demo_data.py` for demo data)\n"
                 "2. `python scripts/train.py`  (add `--demo` for demo data)\n"
                 "3. `python scripts/evaluate.py`, then refresh this page.")
