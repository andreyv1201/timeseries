from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import evaluation_common as common
from evaluation_common import FIRST_TEST_DAY, apply_past_thresholds, summarize


LOCAL_PACKAGES = Path(__file__).resolve().parent / ".xgboost_packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))
from xgboost import XGBRegressor


TRAIN_START = common.TRAIN_START
TRAIN_END = common.TRAIN_END
LAGS = common.LAGS
FEATURES = common.feature_columns()

MODEL_PARAMS = dict(
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=1,
)


def scaled_xy(
    rows: pd.DataFrame, scaler: RobustScaler
) -> tuple[pd.DataFrame, np.ndarray]:

    x = rows[FEATURES].copy()
    for lag in LAGS:
        column = f"lag_{lag}"
        x[column] = scaler.transform(x[[column]].to_numpy()).ravel()
    y = scaler.transform(rows[["y"]].to_numpy()).ravel()
    return x, y


def score_sensor(
    frame: pd.DataFrame, sensor_id: str, max_days: int | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:

    train_mask = (
        (frame.index >= TRAIN_START)
        & (frame.index < TRAIN_END)
        & common.normal_training_rows(frame)
    )
    normal = frame.loc[train_mask]
    if len(normal) < 4 * common.POINTS_PER_DAY:
        raise ValueError(f"[{sensor_id}] not enough normal training rows")


    cut = int(len(normal) * 0.8)
    inner_train, validation = normal.iloc[:cut], normal.iloc[cut:]
    inner_scaler = RobustScaler().fit(inner_train[["y"]].to_numpy())
    x_train, y_train = scaled_xy(inner_train, inner_scaler)
    x_valid, y_valid = scaled_xy(validation, inner_scaler)

    selector = XGBRegressor(
        n_estimators=2000,
        early_stopping_rounds=75,
        **MODEL_PARAMS,
    )
    selector.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    trees = int(selector.best_iteration) + 1


    scaler = RobustScaler().fit(normal[["y"]].to_numpy())
    x_all, y_all = scaled_xy(normal, scaler)
    model = XGBRegressor(n_estimators=trees, **MODEL_PARAMS)
    model.fit(x_all, y_all, verbose=False)

    score_end = frame.index.max() + pd.Timedelta(minutes=10)
    if max_days is not None:
        score_end = min(
            score_end, FIRST_TEST_DAY + pd.Timedelta(days=max_days)
        )
    usable = common.usable_rows(frame)
    score_rows = frame.loc[
        (frame.index >= TRAIN_END) & (frame.index < score_end) & usable
    ]
    x_score, y_score = scaled_xy(score_rows, scaler)
    scores = np.abs(y_score - model.predict(x_score))
    scored = pd.DataFrame(
        {"sensor": sensor_id, "label": score_rows["label"].astype(int), "score": scores},
        index=score_rows.index,
    )
    info = {
        "train_rows": int(len(normal)),
        "inner_train_rows": int(len(inner_train)),
        "validation_rows": int(len(validation)),
        "validation_start": validation.index.min().isoformat(),
        "validation_end": validation.index.max().isoformat(),
        "selected_trees": trees,
        "scaler_center": float(scaler.center_[0]),
        "scaler_scale": float(scaler.scale_[0]),
    }
    print(
        f"[{sensor_id}] normal train={len(normal)}, "
        f"validation={len(validation)}, trees={trees}",
        flush=True,
    )
    return scored, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    values, labels = common.load_panel()
    sensor_ids = [str(sensor) for sensor in values["io_id"].dropna().unique()]
    if args.sensor is not None:
        sensor_ids = [args.sensor]

    report: dict[str, object] = {
        "protocol": {
            "fit": "normal target and normal lags, June 13–21",
            "model_validation": "last 20% of normal train, chronological early stopping",
            "normalization": "sklearn RobustScaler fitted on normal train only",
            "features": FEATURES,
            "threshold": "preceding seven labelled days",
            "test_start": FIRST_TEST_DAY.isoformat(),
        },
        "sensors": {},
    }
    results: list[pd.DataFrame] = []
    for sensor_id in sensor_ids:
        frame = common.make_sensor_frame(values, labels, sensor_id)
        scored, info = score_sensor(frame, sensor_id, args.max_days)
        result = apply_past_thresholds(scored)
        results.append(result)
        report["sensors"][sensor_id] = {
            "fit": info,
            "first_prediction": result.index.min().isoformat(),
            "last_prediction": result.index.max().isoformat(),
            "metrics": summarize(result),
        }
        print(f"[{sensor_id}] XGBoost 1h F1={report['sensors'][sensor_id]['metrics']['1h']['f1']:.6f}")

    report["pooled"] = summarize(pd.concat(results).sort_index())
    print("RESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
