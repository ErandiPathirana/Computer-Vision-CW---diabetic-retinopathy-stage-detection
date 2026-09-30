"""
Small, documented experiments that support the design choices in the report.
All comparisons use the VALIDATION set only; the test set is never used for tuning.

  * run_backbone_comparison  - EfficientNetB0 vs MobileNetV2 vs ResNet50
  * run_hparam_search        - dropout / fine-tuned layers / head learning rate
  * run_imbalance_comparison - class weights vs minority oversampling vs nothing
Every run is short (a few epochs) so the whole set fits inside a Colab session.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List

import keras
import pandas as pd

from . import config as C
from .evaluate import predict_df, quick_val_scores
from .model import parameter_counts
from .train import train_model


def _run(train_df, val_df, name: str, epochs1: int, epochs2: int, **kw) -> Dict:
    """Train one configuration and return its validation scores."""
    t0 = time.time()
    model, _ = train_model(train_df, val_df, epochs1=epochs1, epochs2=epochs2, tag=name, verbose=0, **kw)
    scores = quick_val_scores(val_df["diagnosis"].values, predict_df(model, val_df))
    row = {"run": name, **{k: round(v, 4) for k, v in scores.items()},
           "parameters": parameter_counts(model)["total"], "seconds": round(time.time() - t0)}
    print(row)
    del model
    keras.backend.clear_session()
    return row


def run_backbone_comparison(train_df, val_df, epochs1: int = 5, epochs2: int = 3,
                            backbones=("efficientnetb0", "mobilenetv2", "resnet50")) -> pd.DataFrame:
    C.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"backbone": b, **_run(train_df, val_df, f"bb_{b}", epochs1, epochs2, backbone=b)} for b in backbones]
    df = pd.DataFrame(rows)
    df.to_csv(C.EXPERIMENT_DIR / "backbone_comparison.csv", index=False)
    return df


def run_hparam_search(train_df, val_df, epochs1: int = 4, epochs2: int = 3) -> pd.DataFrame:
    """One-factor-at-a-time search around the baseline (dropout 0.3, 30 layers, head LR 1e-3)."""
    C.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    base = {"dropout": C.DROPOUT, "unfreeze": C.UNFREEZE_LAYERS, "head_lr": C.HEAD_LR}
    trials: List[Dict] = [dict(base)]                                  # baseline first
    trials += [{**base, "dropout": d} for d in (0.2, 0.5)]
    trials += [{**base, "unfreeze": u} for u in (20, 50)]
    trials += [{**base, "head_lr": lr} for lr in (5e-4, 2e-3)]
    rows = []
    for i, t in enumerate(trials):
        rows.append({**t, **_run(train_df, val_df, f"hp_{i}", epochs1, epochs2, **t)})
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df.to_csv(C.EXPERIMENT_DIR / "hyperparameter_search.csv", index=False)
    return df


def best_hparams(search_df: pd.DataFrame) -> Dict:
    """Pick the trial with the best validation macro-F1 (ties -> baseline-like values first)."""
    top = search_df.iloc[0]
    return {"dropout": float(top["dropout"]), "unfreeze": int(top["unfreeze"]), "head_lr": float(top["head_lr"])}


def run_imbalance_comparison(train_df, val_df, epochs1: int = 5, epochs2: int = 3) -> pd.DataFrame:
    C.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"strategy": s, **_run(train_df, val_df, f"imb_{s}", epochs1, epochs2, imbalance=s)}
            for s in ("none", "class_weights", "oversample")]
    df = pd.DataFrame(rows)
    df.to_csv(C.EXPERIMENT_DIR / "imbalance_comparison.csv", index=False)
    (C.EXPERIMENT_DIR / "imbalance_comparison.json").write_text(json.dumps(rows, indent=2))
    return df
