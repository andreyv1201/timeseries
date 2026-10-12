from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

CALIBRATION_DAYS = 7
FIRST_TEST_DAY = pd.Timestamp("2025-06-29")
STEP = pd.Timedelta(minutes=10)
ALARM_POINTS = {"immediate": 1, "30min": 3, "1h": 6, "2h": 12}
POINTS_PER_DAY = 144
LAGS = (144, 288, 432)
TRAIN_START = pd.Timestamp("2025-06-13")
TRAIN_END = pd.Timestamp("2025-06-22")


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.read_csv("mv_cond_temperature.csv")
    labels = pd.read_csv("labled_mv_cond.csv")
    values["datetime"] = pd.to_datetime(values["event_timestamp"], unit="s")
    labels["datetime"] = pd.to_datetime(labels["event_timestamp"], unit="s")
    return values, labels


def make_sensor_frame(
    values: pd.DataFrame, labels: pd.DataFrame, sensor_id: str
) -> pd.DataFrame:
    sensor_values = values.loc[values["io_id"].eq(sensor_id)].sort_values("datetime")
    sensor_labels = labels.loc[labels["io_id"].eq(sensor_id)].sort_values("datetime")
    value_series = (
        pd.to_numeric(sensor_values.set_index("datetime")["value"], errors="coerce")
        .resample("10min")
        .mean()
    )
    label_series = (
        pd.to_numeric(sensor_labels.set_index("datetime")["anomaly"], errors="coerce")
        .resample("10min")
        .max()
        .reindex(value_series.index)
    )
    frame = pd.DataFrame({"y": value_series, "label": label_series})
    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
        frame[f"label_lag_{lag}"] = frame["label"].shift(lag)
    slot_of_day = frame.index.hour * 6 + frame.index.minute // 10
    daily_angle = 2 * np.pi * slot_of_day / POINTS_PER_DAY
    weekly_angle = 2 * np.pi * frame.index.dayofweek / 7
    frame["time_sin"] = np.sin(daily_angle)
    frame["time_cos"] = np.cos(daily_angle)
    frame["dow_sin"] = np.sin(weekly_angle)
    frame["dow_cos"] = np.cos(weekly_angle)
    return frame


def feature_columns() -> list[str]:
    return [
        *(f"lag_{lag}" for lag in LAGS),
        "time_sin", "time_cos", "dow_sin", "dow_cos",
    ]


def usable_rows(frame: pd.DataFrame) -> pd.Series:
    return frame[["y", "label", *feature_columns()]].notna().all(axis=1)


def normal_training_rows(frame: pd.DataFrame) -> pd.Series:
    normal = usable_rows(frame) & frame["label"].eq(0)
    for lag in LAGS:
        normal &= frame[f"label_lag_{lag}"].eq(0)
    return normal


def choose_threshold(
    labels: np.ndarray, scores: np.ndarray, previous: float | None
) -> tuple[float, str]:
    valid = np.isfinite(labels) & np.isfinite(scores)
    labels = np.asarray(labels)[valid].astype(int)
    scores = np.asarray(scores)[valid].astype(float)
    if len(scores) == 0:
        if previous is None:
            raise ValueError("No past calibration scores")
        return previous, "previous:no_scores"

    classes = np.unique(labels)
    if len(classes) == 2:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision[:-1] * recall[:-1] / (
            precision[:-1] + recall[:-1] + 1e-12
        )
        return float(thresholds[int(np.argmax(f1))]), "past_labels:max_f1"
    if classes[0] == 0:
        return float(np.quantile(scores, 0.99)), "past_normal:q99"
    if previous is not None:
        return previous, "previous:positive_only"
    return float(np.quantile(scores, 0.10)), "past_positive:q10"


def confirm_alarm(index: pd.DatetimeIndex, raw: np.ndarray, points: int) -> np.ndarray:
    if points == 1:
        return np.asarray(raw, dtype=int).copy()
    confirmed = np.zeros(len(raw), dtype=int)
    run = 0
    last_time: pd.Timestamp | None = None
    for position, (time, flag) in enumerate(zip(index, raw)):
        if last_time is None or time - last_time != STEP:
            run = 0
        run = run + 1 if flag else 0
        confirmed[position] = int(run >= points)
        last_time = time
    return confirmed


def apply_past_thresholds(
    scored: pd.DataFrame, first_test_day: pd.Timestamp = FIRST_TEST_DAY
) -> pd.DataFrame:

    scored = scored.sort_index()
    days = pd.date_range(first_test_day, scored.index.max().normalize(), freq="D")
    previous: float | None = None
    evaluated: list[pd.DataFrame] = []
    for day in days:
        calibration = scored.loc[
            (scored.index >= day - pd.Timedelta(days=CALIBRATION_DAYS))
            & (scored.index < day)
        ]
        current = scored.loc[(scored.index >= day) & (scored.index < day + pd.Timedelta(days=1))].copy()
        if calibration.empty or current.empty:
            continue
        threshold, source = choose_threshold(
            calibration["label"].to_numpy(), calibration["score"].to_numpy(), previous
        )
        previous = threshold
        current["threshold"] = threshold
        current["threshold_source"] = source
        current["raw_prediction"] = (current["score"] >= threshold).astype(int)
        evaluated.append(current)
    if not evaluated:
        raise ValueError("No test days with past calibration scores")
    result = pd.concat(evaluated).sort_index()
    for name, points in ALARM_POINTS.items():
        result[f"alarm_{name}"] = confirm_alarm(
            result.index, result["raw_prediction"].to_numpy(), points
        )
    return result


def metric_values(result: pd.DataFrame, prediction_column: str) -> dict[str, float | int | None]:
    labels = result["label"].to_numpy(dtype=int)
    scores = result["score"].to_numpy(dtype=float)
    predictions = result[prediction_column].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    two_classes = len(np.unique(labels)) == 2
    return {
        "points": int(len(labels)),
        "normal_points": int((labels == 0).sum()),
        "anomaly_points": int((labels == 1).sum()),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions)) if two_classes else None
        ),
        "roc_auc": float(roc_auc_score(labels, scores)) if two_classes else None,
        "average_precision": (
            float(average_precision_score(labels, scores)) if two_classes else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize(result: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    return {
        duration: metric_values(result, f"alarm_{duration}")
        for duration in ALARM_POINTS
    }
