"""
Storage and cleaning for user-entered historical loss data.

Losses live in a CSV (and a tiny settings JSON) under the data directory, so
they survive restarts and `git pull` deploys. Override the location with the
CRQ_DATA_DIR environment variable.
"""

import json
import os
import tempfile
from typing import Dict, Iterable, Tuple

import pandas as pd

from fair_engine import MIN_EVENTS_FOR_HISTORY, LossHistory

COLUMNS = ["Date", "Scenario", "Loss ($)", "Notes"]
DEFAULT_WINDOW_YEARS = 3

LOSSES_FILE = "loss_history.csv"
SETTINGS_FILE = "settings.json"


def default_data_dir() -> str:
    return os.environ.get("CRQ_DATA_DIR") or os.path.join(os.path.dirname(__file__), "data")


def empty_losses() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.Series(dtype="datetime64[ns]"),
        "Scenario": pd.Series(dtype="object"),
        "Loss ($)": pd.Series(dtype="float64"),
        "Notes": pd.Series(dtype="object"),
    })


def _atomic_write(path: str, write_fn) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            write_fn(f)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_losses(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, LOSSES_FILE)
    if not os.path.exists(path):
        return empty_losses()
    df = pd.read_csv(path)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Loss ($)"] = pd.to_numeric(df["Loss ($)"], errors="coerce")
    df["Notes"] = df["Notes"].fillna("").astype(str)
    return df.reset_index(drop=True)


def clean_losses(df: pd.DataFrame, valid_scenarios: Iterable[str]) -> Tuple[pd.DataFrame, int]:
    """Drop incomplete or invalid rows. Returns (clean_df, number_of_rows_dropped)."""
    valid = set(valid_scenarios)
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Loss ($)"] = pd.to_numeric(out["Loss ($)"], errors="coerce")
    out["Notes"] = out["Notes"].fillna("").astype(str)
    keep = (
        out["Date"].notna()
        & out["Scenario"].isin(valid)
        & out["Loss ($)"].notna()
        & (out["Loss ($)"] > 0)
    )
    dropped = int((~keep).sum())
    out = out[keep].sort_values("Date").reset_index(drop=True)
    return out[COLUMNS], dropped


def save_losses(df: pd.DataFrame, data_dir: str) -> None:
    path = os.path.join(data_dir, LOSSES_FILE)
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    _atomic_write(path, lambda f: out.to_csv(f, index=False))


def load_settings(data_dir: str) -> dict:
    path = os.path.join(data_dir, SETTINGS_FILE)
    settings = {"window_years": DEFAULT_WINDOW_YEARS}
    try:
        with open(path, "r", encoding="utf-8") as f:
            settings.update(json.load(f))
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings: dict, data_dir: str) -> None:
    _atomic_write(os.path.join(data_dir, SETTINGS_FILE), lambda f: json.dump(settings, f, indent=2))


def filter_window(df: pd.DataFrame, window_years: int, as_of=None) -> pd.DataFrame:
    """Rows dated within the last `window_years` years (and not in the future)."""
    end = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=int(window_years))
    return df[(df["Date"] > start) & (df["Date"] <= end)]


def build_histories(
    df: pd.DataFrame, scenario_names: Iterable[str], window_years: int, as_of=None
) -> Dict[str, LossHistory]:
    """One LossHistory per scenario that has enough recorded events in the window."""
    in_window = filter_window(df, window_years, as_of)
    histories = {}
    for name in scenario_names:
        losses = in_window.loc[in_window["Scenario"] == name, "Loss ($)"].to_numpy(dtype=float)
        losses = losses[losses > 0]
        if len(losses) >= MIN_EVENTS_FOR_HISTORY:
            histories[name] = LossHistory(name=name, losses=losses, window_years=float(window_years))
    return histories
