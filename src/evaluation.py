"""Evaluation and bookkeeping helpers shared by the notebook (Part 1 onwards).

Nothing here touches the test partition: callers decide which split to pass in.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import CLASS_NAMES, ROOT

REGISTER_PATH = ROOT / "results" / "experiment_register.csv"
REGISTER_COLUMNS = [
    "id", "part_task", "question", "hypothesis", "changed_variable", "fixed_conditions",
    "split", "seed", "configuration", "metric", "result", "output_path", "notes",
]


# ----------------------------------------------------------------------------- R3 register
def reset_register(prefix: str) -> None:
    """Remove rows whose id starts with `prefix` so re-running a notebook never duplicates rows."""
    if not REGISTER_PATH.exists():
        pd.DataFrame(columns=REGISTER_COLUMNS).to_csv(REGISTER_PATH, index=False)
        return
    df = pd.read_csv(REGISTER_PATH, dtype=str).fillna("")
    df = df[~df["id"].str.startswith(prefix)]
    df.reindex(columns=REGISTER_COLUMNS, fill_value="").to_csv(REGISTER_PATH, index=False)


def log_experiment(**row) -> None:
    """Append one experiment to results/experiment_register.csv (R3)."""
    missing = set(row) - set(REGISTER_COLUMNS)
    if missing:
        raise KeyError(f"unknown register columns: {missing}")
    df = pd.read_csv(REGISTER_PATH, dtype=str).fillna("") if REGISTER_PATH.exists() else pd.DataFrame(columns=REGISTER_COLUMNS)
    df = pd.concat([df, pd.DataFrame([{c: str(row.get(c, "")) for c in REGISTER_COLUMNS}])], ignore_index=True)
    df.to_csv(REGISTER_PATH, index=False)


# ----------------------------------------------------------------------------- fitting
def timed_fit(model, X, y):
    """Fit `model`; return (model, seconds, list of warning messages, e.g. non-convergence)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        model.fit(X, y)
        seconds = time.perf_counter() - t0
    return model, seconds, sorted({str(w.message).split("\n")[0][:80] for w in caught})


# ----------------------------------------------------------------------------- metrics
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (sampling uncertainty of the evaluation set only)."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    """Class-level view: precision, recall, F1, support and the counts behind recall/precision."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(10), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    tp = np.diag(cm)
    predicted = cm.sum(axis=0)
    return pd.DataFrame({
        "class": CLASS_NAMES, "support": s, "correct (TP)": tp,
        "recall": r, "predicted as class": predicted, "precision": p, "F1": f,
    })


def top_confusions(y_true, y_pred, k: int = 6) -> pd.DataFrame:
    """The k most frequent off-diagonal (true -> predicted) confusions."""
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:k], off.shape))[0]
    return pd.DataFrame([
        {"true": CLASS_NAMES[i], "predicted": CLASS_NAMES[j], "count": int(off[i, j]),
         "share of true class": off[i, j] / cm[i].sum()} for i, j in idx
    ])
