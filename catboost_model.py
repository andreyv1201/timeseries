from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
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


DATA_PATH = "mv_cond_temperature.csv"
LABEL_PATH = "labled_mv_cond.csv"
RESAMPLE_RULE = "10min"

POINTS_PER_DAY = 144
LAGS = (144, 288, 432)

TRAIN_START = pd.Timestamp("2025-06-13 00:00:00")
TRAIN_END = pd.Timestamp("2025-06-22 00:00:00")
CALIBRATION_DAYS = 7
FORECAST_DAYS = 1
STEP_DAYS = 1

MIN_ANOMALY_POINTS = 12
NORMAL_ONLY_QUANTILE = 0.99

CATBOOST_PARAMS = dict(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)


@dataclass(frozen=True)
class ThresholdChoice:
    value: float
    source: str


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.read_csv(DATA_PATH)
    labels = pd.read_csv(LABEL_PATH)

    values["datetime"] = pd.to_datetime(values["event_timestamp"], unit="s")
    labels["datetime"] = pd.to_datetime(labels["event_timestamp"], unit="s")
    return values, labels


def make_sensor_frame(
    values: pd.DataFrame, labels: pd.DataFrame, sensor_id: str
) -> pd.DataFrame:
    sensor_values = values.loc[values["io_id"] == sensor_id].sort_values("datetime")
    sensor_labels = labels.loc[labels["io_id"] == sensor_id].sort_values("datetime")

    value_series = (
        sensor_values.set_index("datetime")["value"]
        .apply(pd.to_numeric, errors="coerce")
        .resample(RESAMPLE_RULE)
        .mean()
    )
    label_series = (
        sensor_labels.set_index("datetime")["anomaly"]
        .apply(pd.to_numeric, errors="coerce")
        .resample(RESAMPLE_RULE)
        .max()
        .reindex(value_series.index)
    )

    frame = pd.DataFrame({"y": value_series, "label": label_series})
    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
        frame[f"label_lag_{lag}"] = frame["label"].shift(lag)

    slot_of_day = frame.index.hour * 6 + frame.index.minute // 10
    daily_angle = 2.0 * np.pi * slot_of_day / POINTS_PER_DAY
    weekly_angle = 2.0 * np.pi * frame.index.dayofweek / 7.0

    frame["time_sin"] = np.sin(daily_angle)
    frame["time_cos"] = np.cos(daily_angle)
    frame["dow_sin"] = np.sin(weekly_angle)
    frame["dow_cos"] = np.cos(weekly_angle)
    return frame


def feature_columns() -> list[str]:
    return [
        *(f"lag_{lag}" for lag in LAGS),
        "time_sin",
        "time_cos",
        "dow_sin",
        "dow_cos",
    ]


def usable_rows(frame: pd.DataFrame) -> pd.Series:
    columns = ["y", "label", *feature_columns()]
    return frame[columns].notna().all(axis=1)


def normal_training_rows(frame: pd.DataFrame) -> pd.Series:
    mask = usable_rows(frame) & frame["label"].eq(0)
    for lag in LAGS:
        mask &= frame[f"label_lag_{lag}"].eq(0)
    return mask


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    previous_threshold: float | None,
) -> ThresholdChoice:
    valid = np.isfinite(scores) & np.isfinite(y_true)
    y_true = y_true[valid].astype(int)
    scores = scores[valid].astype(float)
    classes = np.unique(y_true)

    if len(scores) == 0:
        if previous_threshold is None:
            raise ValueError("No past labelled calibration scores")
        return ThresholdChoice(previous_threshold, "previous:no_calibration_scores")

    if len(classes) == 2:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_values = 2 * precision[:-1] * recall[:-1] / (
            precision[:-1] + recall[:-1] + 1e-12
        )
        best = int(np.nanargmax(f1_values))
        return ThresholdChoice(float(thresholds[best]), "past_labels:max_f1")

    if classes[0] == 0:
        threshold = float(np.quantile(scores, NORMAL_ONLY_QUANTILE))
        return ThresholdChoice(threshold, "past_labels:normal_q99")

    if previous_threshold is not None:
        return ThresholdChoice(previous_threshold, "previous:positive_only")

    return ThresholdChoice(float(np.quantile(scores, 0.10)), "past_labels:positive_q10")


def causal_persistence(
    index: pd.DatetimeIndex, raw_prediction: np.ndarray, min_points: int
) -> np.ndarray:
    result = np.zeros(len(raw_prediction), dtype=int)
    run_length = 0
    previous_time: pd.Timestamp | None = None
    expected_step = pd.Timedelta(RESAMPLE_RULE)

    for i, (timestamp, prediction) in enumerate(zip(index, raw_prediction)):
        if previous_time is None or timestamp - previous_time != expected_step:
            run_length = 0
        if prediction == 1:
            run_length += 1
            if run_length >= min_points:
                result[i] = 1
        else:
            run_length = 0
        previous_time = timestamp
    return result


def metric_values(
    y_true: np.ndarray, scores: np.ndarray, prediction: np.ndarray
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    result: dict[str, object] = {
        "points": int(len(y_true)),
        "normal_points": int((y_true == 0).sum()),
        "anomaly_points": int((y_true == 1).sum()),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def fixed_train_sensor(
    frame: pd.DataFrame, sensor_id: str, max_days: int | None = None
) -> pd.DataFrame:
    feature_names = feature_columns()
    fit_range = (frame.index >= TRAIN_START) & (frame.index < TRAIN_END)
    fit_mask = fit_range & normal_training_rows(frame)
    if fit_mask.sum() < 2 * POINTS_PER_DAY:
        raise ValueError(
            f"[{sensor_id}] insufficient normal rows in the fixed train: "
            f"{int(fit_mask.sum())}"
        )

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(frame.loc[fit_mask, feature_names], frame.loc[fit_mask, "y"])
    print(
        f"[{sensor_id}] fixed train={TRAIN_START.date()}.."
        f"{(TRAIN_END - pd.Timedelta(days=1)).date()} "
        f"normal_rows={int(fit_mask.sum())}",
        flush=True,
    )

    first_test_day = TRAIN_END + pd.Timedelta(days=CALIBRATION_DAYS)
    final_test_day = frame.index.max().normalize()
    test_days = pd.date_range(
        first_test_day, final_test_day, freq=f"{STEP_DAYS}D"
    )
    if max_days is not None:
        test_days = test_days[:max_days]

    previous_threshold: float | None = None
    daily_results: list[pd.DataFrame] = []

    for number, test_start in enumerate(test_days, start=1):
        test_end = test_start + pd.Timedelta(days=FORECAST_DAYS)
        calibration_start = test_start - pd.Timedelta(days=CALIBRATION_DAYS)
        calibration_mask = (
            (frame.index >= calibration_start)
            & (frame.index < test_start)
            & usable_rows(frame)
        )
        test_mask = (
            (frame.index >= test_start)
            & (frame.index < test_end)
            & usable_rows(frame)
        )

        if calibration_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        calibration_scores = np.abs(
            frame.loc[calibration_mask, "y"].to_numpy(dtype=float)
            - model.predict(frame.loc[calibration_mask, feature_names])
        )
        threshold_choice = choose_threshold(
            frame.loc[calibration_mask, "label"].to_numpy(dtype=float),
            calibration_scores,
            previous_threshold,
        )
        previous_threshold = threshold_choice.value

        test_scores = np.abs(
            frame.loc[test_mask, "y"].to_numpy(dtype=float)
            - model.predict(frame.loc[test_mask, feature_names])
        )
        daily = pd.DataFrame(
            {
                "sensor": sensor_id,
                "label": frame.loc[test_mask, "label"].to_numpy(dtype=int),
                "score": test_scores,
                "threshold": threshold_choice.value,
                "threshold_source": threshold_choice.source,
                "raw_prediction": (test_scores >= threshold_choice.value).astype(int),
            },
            index=frame.index[test_mask],
        )
        daily_results.append(daily)

        if number == 1 or number % 10 == 0 or number == len(test_days):
            print(
                f"[{sensor_id}] day {number}/{len(test_days)} "
                f"test={test_start.date()} fixed_fit={int(fit_mask.sum())} "
                f"cal={int(calibration_mask.sum())} threshold={threshold_choice.value:.6f} "
                f"source={threshold_choice.source}",
                flush=True,
            )

    if not daily_results:
        raise ValueError(f"[{sensor_id}] no walk-forward predictions")

    result = pd.concat(daily_results).sort_index()
    result["causal_2h_prediction"] = causal_persistence(
        result.index,
        result["raw_prediction"].to_numpy(dtype=int),
        MIN_ANOMALY_POINTS,
    )
    return result


def print_metrics(name: str, metrics: dict[str, object]) -> None:
    print(f"\n{name}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    values, labels = load_panel()
    sensor_ids = [str(value) for value in values["io_id"].dropna().unique()]
    if args.sensor is not None:
        if args.sensor not in sensor_ids:
            raise ValueError(f"Unknown sensor: {args.sensor}")
        sensor_ids = [args.sensor]

    all_results: list[pd.DataFrame] = []
    report: dict[str, object] = {
        "protocol": {
            "train_start": TRAIN_START.isoformat(),
            "train_end_exclusive": TRAIN_END.isoformat(),
            "model_update": "fitted once on fixed normal train and frozen",
            "calibration_days": CALIBRATION_DAYS,
            "forecast_days": FORECAST_DAYS,
            "step_days": STEP_DAYS,
            "threshold": "past labelled calibration days only",
            "features": feature_columns(),
            "minimum_alarm_duration_points": MIN_ANOMALY_POINTS,
        },
        "sensors": {},
    }

    for sensor_id in sensor_ids:
        frame = make_sensor_frame(values, labels, sensor_id)
        result = fixed_train_sensor(frame, sensor_id, args.max_days)
        all_results.append(result)

        raw = metric_values(
            result["label"], result["score"], result["raw_prediction"]
        )
        persistent = metric_values(
            result["label"], result["score"], result["causal_2h_prediction"]
        )
        report["sensors"][sensor_id] = {
            "first_prediction": result.index.min().isoformat(),
            "last_prediction": result.index.max().isoformat(),
            "raw_threshold_metrics": raw,
            "causal_2h_metrics": persistent,
        }
        print_metrics(f"SENSOR {sensor_id} — raw threshold", raw)
        print_metrics(f"SENSOR {sensor_id} — causal 2h alarm", persistent)

    pooled = pd.concat(all_results).sort_index()
    pooled_raw = metric_values(
        pooled["label"], pooled["score"], pooled["raw_prediction"]
    )
    pooled_persistent = metric_values(
        pooled["label"], pooled["score"], pooled["causal_2h_prediction"]
    )
    report["pooled_raw_threshold_metrics"] = pooled_raw
    report["pooled_causal_2h_metrics"] = pooled_persistent

    metric_names = (
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    )
    sensor_reports = list(report["sensors"].values())
    report["macro_raw_threshold_metrics"] = {
        metric: float(
            np.mean(
                [
                    sensor_report["raw_threshold_metrics"][metric]
                    for sensor_report in sensor_reports
                    if sensor_report["raw_threshold_metrics"][metric] is not None
                ]
            )
        )
        for metric in metric_names
    }
    report["macro_causal_2h_metrics"] = {
        metric: float(
            np.mean(
                [
                    sensor_report["causal_2h_metrics"][metric]
                    for sensor_report in sensor_reports
                    if sensor_report["causal_2h_metrics"][metric] is not None
                ]
            )
        )
        for metric in metric_names
    }

    print_metrics("POOLED — raw threshold", pooled_raw)
    print_metrics("POOLED — causal 2h alarm", pooled_persistent)
    print("\nRESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
