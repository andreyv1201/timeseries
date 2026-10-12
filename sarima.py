from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import evaluation_common as common


FIT_DAYS = 7
FIRST_SCORE_DAY = common.FIRST_TEST_DAY - pd.Timedelta(days=common.CALIBRATION_DAYS)
ORDER = (1, 0, 1)
SEASONAL_ORDER = (0, 1, 0, common.POINTS_PER_DAY)


def score_sensor(
    frame: pd.DataFrame, sensor_id: str, max_days: int | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:

    final_day = frame.index.max().normalize()
    if max_days is not None:
        if max_days < 1:
            raise ValueError("max_days must be positive")
        final_day = min(final_day, common.FIRST_TEST_DAY + pd.Timedelta(days=max_days - 1))

    scored_days: list[pd.DataFrame] = []
    unconverged_days: list[str] = []
    warning_days: list[str] = []
    filled_training_points = 0
    for day_number, day in enumerate(pd.date_range(FIRST_SCORE_DAY, final_day, freq="D"), 1):
        train_index = pd.date_range(
            day - pd.Timedelta(days=FIT_DAYS),
            periods=FIT_DAYS * common.POINTS_PER_DAY,
            freq=common.STEP,
        )
        past = frame["y"].reindex(train_index)
        filled_training_points += int(past.isna().sum())

        past = past.interpolate(method="time", limit_direction="both")
        if past.isna().any():
            raise ValueError(f"[{sensor_id}] insufficient history before {day.date()}")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SARIMAX(
                past.to_numpy(dtype=float),
                order=ORDER,
                seasonal_order=SEASONAL_ORDER,
                trend="c",
                simple_differencing=True,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=100)
        if not fit.mle_retvals.get("converged", False):
            unconverged_days.append(day.date().isoformat())
        if caught:
            warning_days.append(day.date().isoformat())


        forecast_result = fit.get_forecast(steps=common.POINTS_PER_DAY)
        difference_forecast = np.asarray(forecast_result.predicted_mean, dtype=float)
        forecast_std = np.asarray(forecast_result.se_mean, dtype=float)
        if not np.isfinite(forecast_std).all() or (forecast_std <= 0).any():
            raise ValueError(f"[{sensor_id}] invalid forecast uncertainty on {day.date()}")
        forecast = past.to_numpy(dtype=float)[-common.POINTS_PER_DAY:] + difference_forecast
        day_index = pd.date_range(day, periods=common.POINTS_PER_DAY, freq=common.STEP)
        actual = frame["y"].reindex(day_index)
        labels = frame["label"].reindex(day_index)

        comparable = common.usable_rows(frame).reindex(day_index).fillna(False)
        valid = comparable & np.isfinite(forecast)
        if valid.any():
            absolute_error = np.abs(
                actual.loc[valid].to_numpy(dtype=float)
                - forecast[np.asarray(valid, dtype=bool)]
            )
            scored_days.append(
                pd.DataFrame(
                    {
                        "sensor": sensor_id,
                        "label": labels.loc[valid].to_numpy(dtype=int),
                        "score": absolute_error / forecast_std[np.asarray(valid, dtype=bool)],
                        "absolute_error": absolute_error,
                    },
                    index=day_index[valid],
                )
            )
        if day_number == 1 or day_number % 15 == 0 or day == final_day:
            print(f"[{sensor_id}] SARIMA day {day_number}: {day.date()}", flush=True)

    if not scored_days:
        raise ValueError(f"[{sensor_id}] no forecast scores")
    info = {
        "fit_days": FIT_DAYS,
        "first_train_start": (FIRST_SCORE_DAY - pd.Timedelta(days=FIT_DAYS)).isoformat(),
        "first_score_day": FIRST_SCORE_DAY.isoformat(),
        "fits": int((final_day - FIRST_SCORE_DAY).days + 1),
        "filled_training_points_across_fits": filled_training_points,
        "unconverged_days": unconverged_days,
        "warning_days": warning_days,
    }
    return pd.concat(scored_days).sort_index(), info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--max-days", type=int, default=None,
                        help="Test days to score, for a quick smoke test")
    args = parser.parse_args()

    values, labels = common.load_panel()
    sensor_ids = [str(sensor) for sensor in values["io_id"].dropna().unique()]
    if args.sensor is not None:
        if args.sensor not in sensor_ids:
            raise ValueError(f"Unknown sensor: {args.sensor}")
        sensor_ids = [args.sensor]

    report: dict[str, object] = {
        "protocol": {
            "model": "statsmodels SARIMAX(1,0,1)(0,1,0)[144]",
            "fit": "new fit each day, preceding seven calendar days of observations",
            "forecast": "next 144 ten-minute values fixed before that day's observations",
            "model_labels": "none",
            "score": "absolute forecast error divided by SARIMA forecast standard deviation",
            "threshold": "previous seven labelled days: max F1 with both classes; q99 if only normal; retain previous threshold if only anomalies (q10 if none yet)",
            "calibrated_score": "daily standardized score / that day's past-only threshold; used for a separate ROC-AUC/AP, not for selecting the alarm",
            "test_start": common.FIRST_TEST_DAY.isoformat(),
            "test_filter": "same complete rows as lag-based models",
        },
        "sensors": {},
    }
    results: list[pd.DataFrame] = []
    absolute_results: list[pd.DataFrame] = []
    calibrated_results: list[pd.DataFrame] = []
    for sensor_id in sensor_ids:
        frame = common.make_sensor_frame(values, labels, sensor_id)
        scored, info = score_sensor(frame, sensor_id, args.max_days)
        result = common.apply_past_thresholds(scored)
        absolute_result = common.apply_past_thresholds(
            scored.assign(score=scored["absolute_error"])
        )
        results.append(result)
        absolute_results.append(absolute_result)
        if (result["threshold"] <= 0).any():
            raise ValueError(f"[{sensor_id}] nonpositive calibration threshold")
        calibrated = result.assign(score=result["score"] / result["threshold"])
        calibrated_results.append(calibrated)
        report["sensors"][sensor_id] = {
            "fit": info,
            "first_prediction": result.index.min().isoformat(),
            "last_prediction": result.index.max().isoformat(),
            "metrics": common.summarize(result),
            "calibrated_score_metrics": common.summarize(calibrated),
            "absolute_error_metrics": common.summarize(absolute_result),
        }
        print(f"[{sensor_id}] 1h F1={report['sensors'][sensor_id]['metrics']['1h']['f1']:.6f}")

    report["pooled"] = common.summarize(pd.concat(results).sort_index())
    report["pooled_calibrated_score"] = common.summarize(
        pd.concat(calibrated_results).sort_index()
    )
    report["pooled_absolute_error"] = common.summarize(
        pd.concat(absolute_results).sort_index()
    )
    print("RESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
