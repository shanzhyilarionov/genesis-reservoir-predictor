from pathlib import Path
from datetime import datetime, timezone

import joblib


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def save_artifact(artifact, file_path):
    ensure_parent_dir(file_path)
    joblib.dump(artifact, file_path)


def load_artifact(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    return joblib.load(path)


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def risk_level(risk):
    if risk < 0.3:
        return "Low"
    if risk < 0.7:
        return "Medium"
    return "High"


def format_percent(value):
    return f"{value * 100:.2f}%"


def require_artifact_keys(artifact, keys):
    missing = [key for key in keys if key not in artifact]

    if missing:
        raise ValueError(f"Invalid model artifact. Missing keys: {missing}")