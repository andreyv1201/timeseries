from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import stumpy
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
WINDOW_SIZE = 3 * POINTS_PER_DAY
PAST_DAYS = 21
CALIBRATION_DAYS = 7
DEVELOPMENT_DAYS = 21
FORECAST_DAYS = 1
STEP_DAYS = 1
NORMAL_ONLY_QUANTILE = 0.99
MIN_ANOMALY_POINTS = 36


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
    return pd.DataFrame({"value": value_series, "label": label_series})


def fill_past(series: pd.Series) -> pd.Series:


    result = series.interpolate(method="time", limit_direction="both")
    if result.isna().any():
        raise ValueError("Past Matrix Profile interval could not be filled")
    return result.astype(float)


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


def metric_values(
    y_true: np.ndarray, scores: np.ndarray, prediction: np.ndarray
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2

    return {
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


def daily_matrix_profile_scores(
    frame: pd.DataFrame, sensor_id: str, max_days: int | None = None
) -> pd.DataFrame:
    first_complete_day = frame.index.min().normalize() + pd.Timedelta(days=1)
    first_forecast_day = first_complete_day + pd.Timedelta(days=PAST_DAYS)
    final_forecast_day = frame.index.max().normalize()
    forecast_days = pd.date_range(
        first_forecast_day, final_forecast_day, freq=f"{STEP_DAYS}D"
    )
    if max_days is not None:
        forecast_days = forecast_days[: CALIBRATION_DAYS + max_days]

    context_points = WINDOW_SIZE - 1
    daily_results: list[pd.DataFrame] = []

    for number, forecast_start in enumerate(forecast_days, start=1):
        forecast_end = forecast_start + pd.Timedelta(days=FORECAST_DAYS)
        past_start = forecast_start - pd.Timedelta(days=PAST_DAYS)
        expected_past_index = pd.date_range(
            past_start,
            forecast_start - pd.Timedelta(RESAMPLE_RULE),
            freq=RESAMPLE_RULE,
        )
        past = fill_past(frame["value"].reindex(expected_past_index))


        reference = past.iloc[:-context_points].to_numpy(dtype=np.float64)
        context = past.iloc[-context_points:]

        test_index = pd.date_range(
            forecast_start,
            forecast_end - pd.Timedelta(RESAMPLE_RULE),
            freq=RESAMPLE_RULE,
        )
        actual = frame["value"].reindex(test_index)
        labels = frame["label"].reindex(test_index)


        query = pd.concat([context, actual]).ffill()
        if query.isna().any():
            raise ValueError("Causal Matrix Profile query contains NaN")

        profile = stumpy.stump(
            query.to_numpy(dtype=np.float64),
            m=WINDOW_SIZE,
            T_B=reference,
            ignore_trivial=False,
        )
        scores_all = np.asarray(profile[:, 0], dtype=float)
        if len(scores_all) != len(test_index):
            raise AssertionError("One Matrix Profile score per test point was expected")

        valid = actual.notna() & labels.notna() & np.isfinite(scores_all)
        valid_array = np.asarray(valid, dtype=bool)
        daily_results.append(
            pd.DataFrame(
                {
                    "sensor": sensor_id,
                    "label": labels.loc[valid].to_numpy(dtype=int),
                    "score": scores_all[valid_array],
                    "forecast_day": forecast_start,
                },
                index=test_index[valid],
            )
        )

        if number == 1 or number % 10 == 0 or number == len(forecast_days):
            print(
                f"[{sensor_id}] forecast {number}/{len(forecast_days)} "
                f"day={forecast_start.date()} reference={len(reference)} "
                f"query_scores={len(scores_all)}",
                flush=True,
            )

    if not daily_results:
        raise ValueError(f"[{sensor_id}] no daily Matrix Profile scores")
    return pd.concat(daily_results).sort_index()


def apply_past_thresholds(
    scored: pd.DataFrame, min_anomaly_points: int
) -> pd.DataFrame:
    forecast_days = pd.DatetimeIndex(
        scored["forecast_day"].drop_duplicates()
    ).sort_values()
    previous_threshold: float | None = None
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
        current["raw_prediction"] = (
            current["score"].to_numpy(dtype=float) >= threshold.value
        ).astype(int)
        evaluated_days.append(current)

    if not evaluated_days:
        raise ValueError("Not enough Matrix Profile forecasts after calibration")

    result = pd.concat(evaluated_days).sort_index()
    raw = result["raw_prediction"].astype(bool)
    if min_anomaly_points == 1:
        result["prediction"] = raw.astype(int)
    else:
        false_groups = (~raw).cumsum()
        run_length = raw.groupby(false_groups).cumsum()
        result["prediction"] = (run_length >= min_anomaly_points).astype(int)
    return result


def print_metrics(name: str, metrics: dict[str, object]) -> None:
    print(f"\n{name}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def macro_metrics(sensor_metrics: list[dict[str, object]]) -> dict[str, float]:
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
        values = [metrics[name] for metrics in sensor_metrics if metrics[name] is not None]
        if values:
            result[name] = float(np.mean(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--min-anomaly-points", type=int, default=MIN_ANOMALY_POINTS
    )
    args = parser.parse_args()
    if args.min_anomaly_points < 1:
        raise ValueError("--min-anomaly-points must be positive")

    values, labels = load_panel()
    sensor_ids = [str(value) for value in values["io_id"].dropna().unique()]
    if args.sensor is not None:
        if args.sensor not in sensor_ids:
            raise ValueError(f"Unknown sensor: {args.sensor}")
        sensor_ids = [args.sensor]

    all_results: list[pd.DataFrame] = []
    all_holdouts: list[pd.DataFrame] = []
    report: dict[str, object] = {
        "protocol": {
            "past_days": PAST_DAYS,
            "window_points": WINDOW_SIZE,
            "window_days": WINDOW_SIZE / POINTS_PER_DAY,
            "distance": "z_normalized",
            "initial_calibration_days": CALIBRATION_DAYS,
            "development_days_for_confirmation": DEVELOPMENT_DAYS,
            "threshold_history_days": CALIBRATION_DAYS,
            "forecast_days": FORECAST_DAYS,
            "step_days": STEP_DAYS,
            "threshold": "preceding labelled out-of-sample days only",
            "score_alignment": "window endpoint; causal",
            "reference_overlap": "excluded",
            "minimum_consecutive_anomaly_points": args.min_anomaly_points,
        },
        "sensors": {},
    }

    for sensor_id in sensor_ids:
        frame = make_sensor_frame(values, labels, sensor_id)
        scored = daily_matrix_profile_scores(frame, sensor_id, args.max_days)
        result = apply_past_thresholds(scored, args.min_anomaly_points)
        all_results.append(result)

        metrics = metric_values(
            result["label"], result["score"], result["prediction"]
        )
        result_days = pd.DatetimeIndex(
            result["forecast_day"].drop_duplicates()
        ).sort_values()
        holdout_days = result_days[DEVELOPMENT_DAYS:]
        holdout = result.loc[result["forecast_day"].isin(holdout_days)]
        holdout_metrics = (
            metric_values(
                holdout["label"], holdout["score"], holdout["prediction"]
            )
            if len(holdout)
            else None
        )
        if len(holdout):
            all_holdouts.append(holdout)

        report["sensors"][sensor_id] = {
            "first_prediction": result.index.min().isoformat(),
            "last_prediction": result.index.max().isoformat(),
            "metrics": metrics,
            "late_holdout_start": (
                holdout.index.min().isoformat() if len(holdout) else None
            ),
            "late_holdout_metrics": holdout_metrics,
        }
        print_metrics(f"SENSOR {sensor_id}", metrics)
        if holdout_metrics is not None:
            print_metrics(f"SENSOR {sensor_id} LATE HOLDOUT", holdout_metrics)

    pooled = pd.concat(all_results).sort_index()
    pooled_metrics = metric_values(
        pooled["label"], pooled["score"], pooled["prediction"]
    )
    sensor_metrics = [entry["metrics"] for entry in report["sensors"].values()]
    report["macro_metrics"] = macro_metrics(sensor_metrics)
    report["pooled_metrics"] = pooled_metrics
    print_metrics("POOLED", pooled_metrics)

    holdout_sensor_metrics = [
        entry["late_holdout_metrics"]
        for entry in report["sensors"].values()
        if entry["late_holdout_metrics"] is not None
    ]
    if all_holdouts:
        pooled_holdout = pd.concat(all_holdouts).sort_index()
        pooled_holdout_metrics = metric_values(
            pooled_holdout["label"],
            pooled_holdout["score"],
            pooled_holdout["prediction"],
        )
        report["late_holdout_macro_metrics"] = macro_metrics(
            holdout_sensor_metrics
        )
        report["late_holdout_pooled_metrics"] = pooled_holdout_metrics
        print_metrics("LATE HOLDOUT POOLED", pooled_holdout_metrics)

    print("\nRESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
