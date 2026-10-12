from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

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


DATA_PATH = "mv_cond_temperature.csv"
LABEL_PATH = "labled_mv_cond.csv"
RESAMPLE_RULE = "10min"

POINTS_PER_DAY = 144
SEASONAL_PERIOD = POINTS_PER_DAY
TRAIN_DAYS = 21
CALIBRATION_DAYS = 7
FORECAST_DAYS = 1
STEP_DAYS = 1

ALPHA = 0.25
BETA = 0.0
GAMMA = 0.25
CLIP_RANGE = (0.0, 35.0)

MIN_ANOMALY_POINTS = 72
ONE_HOUR_ALARM_POINTS = 6
PERSISTENCE_CANDIDATES = (1, 12, 36, 72)
NORMAL_ONLY_QUANTILE = 0.99


@dataclass(frozen=True)
class ThresholdChoice:
    value: float
    source: str


class ETSAnomalyDetector:
    def holt_winters_forecast(
        self,
        series: np.ndarray,
        steps: int = POINTS_PER_DAY,
        seasonal_period: int = SEASONAL_PERIOD,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
        clip_range: tuple[float, float] | None = CLIP_RANGE,
    ) -> np.ndarray:
        values = np.asarray(series, dtype=float)
        n = len(values)
        if n < 2 * seasonal_period:
            raise ValueError("At least two seasonal periods are required")
        if not np.isfinite(values).all():
            raise ValueError("ETS training series contains non-finite values")

        level = np.zeros(n, dtype=float)
        trend = np.zeros(n, dtype=float)
        seasonal = np.zeros(seasonal_period, dtype=float)

        level[0] = float(np.mean(values[:seasonal_period]))
        if beta == 0.0:


            trend[0] = 0.0
        else:
            first_period_mean = float(np.mean(values[:seasonal_period]))
            second_period_mean = float(
                np.mean(values[seasonal_period : 2 * seasonal_period])
            )
            trend[0] = (
                second_period_mean - first_period_mean
            ) / seasonal_period

        for position in range(seasonal_period):
            indices = np.arange(position, n, seasonal_period)
            seasonal[position] = float(np.mean(values[indices] - level[0]))

        for t in range(1, n):
            season = seasonal[t % seasonal_period]
            previous_level = level[t - 1]
            previous_trend = trend[t - 1]

            level[t] = alpha * (values[t] - season) + (1 - alpha) * (
                previous_level + previous_trend
            )
            trend[t] = beta * (level[t] - previous_level) + (
                1 - beta
            ) * previous_trend
            seasonal[t % seasonal_period] = gamma * (
                values[t] - level[t]
            ) + (1 - gamma) * season

        forecast = np.empty(steps, dtype=float)
        for step in range(steps):
            seasonal_index = (n + step) % seasonal_period
            value = (
                level[-1]
                + (step + 1) * trend[-1]
                + seasonal[seasonal_index]
            )
            if clip_range is not None:
                value = float(np.clip(value, clip_range[0], clip_range[1]))
            forecast[step] = value
        return forecast


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
    return pd.DataFrame({"value": value_series, "label": label_series})


def fill_past_training_window(series: pd.Series) -> np.ndarray:


    filled = series.interpolate(method="time", limit_direction="both")
    if filled.isna().any():
        raise ValueError("ETS training window could not be filled")
    return filled.to_numpy(dtype=float)


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
        return ThresholdChoice(
            float(np.quantile(scores, NORMAL_ONLY_QUANTILE)),
            "past_labels:normal_q99",
        )

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


def choose_persistence_points(
    index: pd.DatetimeIndex,
    y_true: np.ndarray,
    raw_prediction: np.ndarray,
    previous_points: int | None,
) -> int:
    y_true = np.asarray(y_true, dtype=int)
    raw_prediction = np.asarray(raw_prediction, dtype=int)
    classes = np.unique(y_true)

    if len(classes) == 1 and classes[0] == 1 and previous_points is not None:
        return previous_points

    candidates: list[tuple[float, int, int]] = []
    for points in PERSISTENCE_CANDIDATES:
        prediction = causal_persistence(index, raw_prediction, points)
        if len(classes) == 2:
            quality = float(f1_score(y_true, prediction, zero_division=0))
            false_positives = int(((prediction == 1) & (y_true == 0)).sum())
        else:


            false_positives = int(prediction.sum())
            quality = -float(false_positives)
        candidates.append((quality, -false_positives, -points))

    best = max(range(len(candidates)), key=candidates.__getitem__)
    return int(PERSISTENCE_CANDIDATES[best])


def metric_values(
    y_true: np.ndarray, scores: np.ndarray, prediction: np.ndarray
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2

    result: dict[str, object] = {
        "points": int(len(y_true)),
        "normal_points": int((y_true == 0).sum()),
        "anomaly_points": int((y_true == 1).sum()),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, prediction))
            if two_classes
            else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": float(roc_auc_score(y_true, scores)) if two_classes else None,
        "average_precision": (
            float(average_precision_score(y_true, scores)) if two_classes else None
        ),
    }
    return result


def daily_out_of_sample_scores(
    frame: pd.DataFrame, sensor_id: str, max_days: int | None = None
) -> pd.DataFrame:
    first_complete_day = frame.index.min().normalize() + pd.Timedelta(days=1)
    first_forecast_day = first_complete_day + pd.Timedelta(days=TRAIN_DAYS)
    final_forecast_day = frame.index.max().normalize()
    forecast_days = pd.date_range(
        first_forecast_day, final_forecast_day, freq=f"{STEP_DAYS}D"
    )
    if max_days is not None:
        required = CALIBRATION_DAYS + max_days
        forecast_days = forecast_days[:required]

    detector = ETSAnomalyDetector()
    daily_results: list[pd.DataFrame] = []

    for number, forecast_start in enumerate(forecast_days, start=1):
        forecast_end = forecast_start + pd.Timedelta(days=FORECAST_DAYS)
        train_start = forecast_start - pd.Timedelta(days=TRAIN_DAYS)

        expected_train_index = pd.date_range(
            train_start,
            forecast_start - pd.Timedelta(RESAMPLE_RULE),
            freq=RESAMPLE_RULE,
        )
        train_series = frame["value"].reindex(expected_train_index)
        train_values = fill_past_training_window(train_series)
        forecast = detector.holt_winters_forecast(
            train_values,
            steps=FORECAST_DAYS * POINTS_PER_DAY,
        )

        test_mask = (
            (frame.index >= forecast_start)
            & (frame.index < forecast_end)
            & frame["value"].notna()
            & frame["label"].notna()
        )
        actual = frame.loc[test_mask, "value"].to_numpy(dtype=float)
        positions = (
            (frame.index[test_mask] - forecast_start) / pd.Timedelta(RESAMPLE_RULE)
        ).astype(int)
        scores = np.abs(actual - forecast[np.asarray(positions, dtype=int)])

        daily_results.append(
            pd.DataFrame(
                {
                    "sensor": sensor_id,
                    "label": frame.loc[test_mask, "label"].to_numpy(dtype=int),
                    "score": scores,
                    "forecast_day": forecast_start,
                },
                index=frame.index[test_mask],
            )
        )
        if number == 1 or number % 10 == 0 or number == len(forecast_days):
            print(
                f"[{sensor_id}] forecast {number}/{len(forecast_days)} "
                f"day={forecast_start.date()} train_points={len(train_values)}",
                flush=True,
            )

    if not daily_results:
        raise ValueError(f"[{sensor_id}] no daily ETS forecasts")
    return pd.concat(daily_results).sort_index()


def apply_past_thresholds(scored: pd.DataFrame) -> pd.DataFrame:
    forecast_days = pd.DatetimeIndex(scored["forecast_day"].drop_duplicates()).sort_values()
    previous_threshold: float | None = None
    previous_persistence_points: int | None = None
    evaluated_days: list[pd.DataFrame] = []

    for position in range(CALIBRATION_DAYS, len(forecast_days)):
        current_day = forecast_days[position]
        past_days = forecast_days[position - CALIBRATION_DAYS : position]
        calibration = scored.loc[scored["forecast_day"].isin(past_days)]
        current = scored.loc[scored["forecast_day"].eq(current_day)].copy()

        threshold = choose_threshold(
            calibration["label"].to_numpy(dtype=float),
            calibration["score"].to_numpy(dtype=float),
            previous_threshold,
        )
        previous_threshold = threshold.value
        current["threshold"] = threshold.value
        current["threshold_source"] = threshold.source
        calibration_raw = (
            calibration["score"].to_numpy(dtype=float) >= threshold.value
        ).astype(int)
        current_raw = (
            current["score"].to_numpy(dtype=float) >= threshold.value
        ).astype(int)
        current["raw_prediction"] = current_raw

        persistence_points = choose_persistence_points(
            calibration.index,
            calibration["label"].to_numpy(dtype=int),
            calibration_raw,
            previous_persistence_points,
        )
        previous_persistence_points = persistence_points

        combined_index = calibration.index.append(current.index)
        combined_raw = np.concatenate([calibration_raw, current_raw])
        combined_prediction = causal_persistence(
            combined_index, combined_raw, persistence_points
        )
        current["adaptive_prediction"] = combined_prediction[-len(current) :]
        current["persistence_points"] = persistence_points
        evaluated_days.append(current)

    if not evaluated_days:
        raise ValueError("Not enough daily forecasts after threshold calibration")

    result = pd.concat(evaluated_days).sort_index()
    result["causal_1h_prediction"] = causal_persistence(
        result.index,
        result["raw_prediction"].to_numpy(dtype=int),
        ONE_HOUR_ALARM_POINTS,
    )
    result["causal_12h_prediction"] = causal_persistence(
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


def macro_metrics(
    sensor_reports: list[dict[str, dict[str, object]]], result_name: str
) -> dict[str, float]:
    names = (
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    )
    result: dict[str, float] = {}
    for name in names:
        values = [
            report[result_name][name]
            for report in sensor_reports
            if report[result_name][name] is not None
        ]
        if values:
            result[name] = float(np.mean(values))
    return result


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
            "train_days": TRAIN_DAYS,
            "initial_calibration_days": CALIBRATION_DAYS,
            "threshold_history_days": CALIBRATION_DAYS,
            "forecast_days": FORECAST_DAYS,
            "step_days": STEP_DAYS,
            "threshold": "preceding labelled out-of-sample days only",
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "minimum_alarm_duration_points": MIN_ANOMALY_POINTS,
            "one_hour_alarm_points": ONE_HOUR_ALARM_POINTS,
            "adaptive_duration_candidates_points": PERSISTENCE_CANDIDATES,
        },
        "sensors": {},
    }

    for sensor_id in sensor_ids:
        frame = make_sensor_frame(values, labels, sensor_id)
        scored = daily_out_of_sample_scores(frame, sensor_id, args.max_days)
        result = apply_past_thresholds(scored)
        all_results.append(result)

        raw = metric_values(
            result["label"], result["score"], result["raw_prediction"]
        )
        one_hour = metric_values(
            result["label"], result["score"], result["causal_1h_prediction"]
        )
        persistent = metric_values(
            result["label"], result["score"], result["causal_12h_prediction"]
        )
        adaptive = metric_values(
            result["label"], result["score"], result["adaptive_prediction"]
        )
        report["sensors"][sensor_id] = {
            "first_prediction": result.index.min().isoformat(),
            "last_prediction": result.index.max().isoformat(),
            "raw_threshold_metrics": raw,
            "causal_1h_metrics": one_hour,
            "causal_12h_metrics": persistent,
            "adaptive_persistence_metrics": adaptive,
        }
        print_metrics(f"SENSOR {sensor_id} — raw threshold", raw)
        print_metrics(f"SENSOR {sensor_id} — causal 1h alarm", one_hour)
        print_metrics(f"SENSOR {sensor_id} — causal 12h alarm", persistent)
        print_metrics(f"SENSOR {sensor_id} — past-selected duration", adaptive)

    pooled = pd.concat(all_results).sort_index()
    pooled_raw = metric_values(
        pooled["label"], pooled["score"], pooled["raw_prediction"]
    )
    pooled_one_hour = metric_values(
        pooled["label"], pooled["score"], pooled["causal_1h_prediction"]
    )
    pooled_persistent = metric_values(
        pooled["label"], pooled["score"], pooled["causal_12h_prediction"]
    )
    pooled_adaptive = metric_values(
        pooled["label"], pooled["score"], pooled["adaptive_prediction"]
    )
    report["pooled_raw_threshold_metrics"] = pooled_raw
    report["pooled_causal_1h_metrics"] = pooled_one_hour
    report["pooled_causal_12h_metrics"] = pooled_persistent
    report["pooled_adaptive_persistence_metrics"] = pooled_adaptive

    sensor_reports = list(report["sensors"].values())
    report["macro_raw_threshold_metrics"] = macro_metrics(
        sensor_reports, "raw_threshold_metrics"
    )
    report["macro_causal_1h_metrics"] = macro_metrics(
        sensor_reports, "causal_1h_metrics"
    )
    report["macro_causal_12h_metrics"] = macro_metrics(
        sensor_reports, "causal_12h_metrics"
    )
    report["macro_adaptive_persistence_metrics"] = macro_metrics(
        sensor_reports, "adaptive_persistence_metrics"
    )

    print_metrics("POOLED — raw threshold", pooled_raw)
    print_metrics("POOLED — causal 1h alarm", pooled_one_hour)
    print_metrics("POOLED — causal 12h alarm", pooled_persistent)
    print_metrics("POOLED — past-selected duration", pooled_adaptive)
    print("\nRESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
