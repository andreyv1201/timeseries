from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import QuantileTransformer, RobustScaler

import catboost_model as baseline
import srcnn
from evaluation_common import (
    ALARM_POINTS,
    FIRST_TEST_DAY,
    apply_past_thresholds,
    choose_threshold,
    confirm_alarm,
    summarize,
)


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["hour"] = frame.index.hour
    frame["dow"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    return frame


def add_sr(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    sr_frame = pd.DataFrame({"value": frame["y"]}, index=frame.index)
    sr_cache = srcnn.precompute_feature_cache(sr_frame)
    frame["sr"] = sr_cache.real[:, srcnn.WINDOW - 1]
    frame["sr_lag_1"] = frame["sr"].shift(1)
    return frame


def features_for(variant: str) -> list[str]:
    if variant == "original":
        return ["lag_144", "lag_288", "lag_432", "hour", "dow", "month"]
    features = baseline.feature_columns()
    if variant == "cyclic_sr_lag1":
        features = [*features, "sr_lag_1"]
    return features


def score_model(
    frame: pd.DataFrame,
    sensor_id: str,
    variant: str,
    max_days: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    features = features_for(variant)
    normal = (
        (frame.index >= baseline.TRAIN_START)
        & (frame.index < baseline.TRAIN_END)
        & baseline.normal_training_rows(frame)
        & frame[features].notna().all(axis=1)
    )
    if int(normal.sum()) < 2 * baseline.POINTS_PER_DAY:
        raise ValueError(f"[{sensor_id}] insufficient normal training rows")
    fit_x = frame.loc[normal, features].copy()
    fit_y = frame.loc[normal, "y"].to_numpy(dtype=float)
    scaler: RobustScaler | None = None
    if variant == "cyclic_scaled":
        scaler = RobustScaler().fit(fit_y.reshape(-1, 1))
        fit_y = scaler.transform(fit_y.reshape(-1, 1)).ravel()
        for lag in baseline.LAGS:
            column = f"lag_{lag}"
            fit_x[column] = scaler.transform(fit_x[[column]].to_numpy()).ravel()
    model = CatBoostRegressor(**baseline.CATBOOST_PARAMS)
    model.fit(fit_x, fit_y)

    score_end = frame.index.max() + pd.Timedelta(minutes=10)
    if max_days is not None:
        score_end = min(score_end, FIRST_TEST_DAY + pd.Timedelta(days=max_days))
    score_mask = (
        (frame.index >= baseline.TRAIN_END)
        & (frame.index < score_end)
        & baseline.usable_rows(frame)
        & frame[features].notna().all(axis=1)
    )
    selected = frame.loc[score_mask]
    score_x = selected[features].copy()
    score_y = selected["y"].to_numpy(dtype=float)
    if scaler is not None:
        score_y = scaler.transform(score_y.reshape(-1, 1)).ravel()
        for lag in baseline.LAGS:
            column = f"lag_{lag}"
            score_x[column] = scaler.transform(score_x[[column]].to_numpy()).ravel()
    residual = np.abs(score_y - model.predict(score_x))
    scored = pd.DataFrame(
        {
            "sensor": sensor_id,
            "label": selected["label"].astype(int),
            "score": residual,
            "residual": residual,
        },
        index=selected.index,
    )
    if "sr" in frame:
        scored["sr"] = selected["sr"].to_numpy(dtype=float)
    info = {"features": features, "normal_train_rows": int(normal.sum())}
    if scaler is not None:
        info["train_scaler_center"] = float(scaler.center_[0])
        info["train_scaler_scale"] = float(scaler.scale_[0])
    print(f"[{sensor_id}] CatBoost {variant}, normal train={int(normal.sum())}", flush=True)
    return scored, info


def fuse_scores(scored: pd.DataFrame, sr_weight: float) -> pd.DataFrame:

    days = pd.date_range(FIRST_TEST_DAY, scored.index.max().normalize(), freq="D")
    previous_threshold: float | None = None
    evaluated: list[pd.DataFrame] = []
    for day in days:
        calibration = scored.loc[
            (scored.index >= day - pd.Timedelta(days=7)) & (scored.index < day)
        ].copy()
        current = scored.loc[
            (scored.index >= day) & (scored.index < day + pd.Timedelta(days=1))
        ].copy()
        calibration = calibration.dropna(subset=["residual", "sr"])
        current = current.dropna(subset=["residual", "sr"])
        if calibration.empty or current.empty:
            continue

        normal_reference = calibration.loc[calibration["label"].eq(0)]
        reference = normal_reference if len(normal_reference) >= 50 else calibration
        scaler = QuantileTransformer(
            n_quantiles=min(1000, len(reference)),
            output_distribution="uniform",
            random_state=42,
        )
        columns = ["residual", "sr"]
        scaler.fit(reference[columns].to_numpy())
        past_scaled = scaler.transform(calibration[columns].to_numpy())
        today_scaled = scaler.transform(current[columns].to_numpy())
        calibration["score"] = (
            (1 - sr_weight) * past_scaled[:, 0] + sr_weight * past_scaled[:, 1]
        )
        current["score"] = (
            (1 - sr_weight) * today_scaled[:, 0] + sr_weight * today_scaled[:, 1]
        )
        threshold, source = choose_threshold(
            calibration["label"].to_numpy(),
            calibration["score"].to_numpy(),
            previous_threshold,
        )
        previous_threshold = threshold
        current["threshold"] = threshold
        current["threshold_source"] = source
        current["raw_prediction"] = (current["score"] >= threshold).astype(int)
        evaluated.append(current)

    if not evaluated:
        raise ValueError("No days available for SR fusion")
    result = pd.concat(evaluated).sort_index()
    for name, points in ALARM_POINTS.items():
        result[f"alarm_{name}"] = confirm_alarm(
            result.index, result["raw_prediction"].to_numpy(), points
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    values, labels = baseline.load_panel()
    sensor_ids = [str(sensor) for sensor in values["io_id"].dropna().unique()]
    if args.sensor is not None:
        sensor_ids = [args.sensor]
    variants = (
        "original",
        "cyclic",
        "cyclic_scaled",
        "cyclic_sr_lag1",
        "cyclic_sr_fusion25",
        "cyclic_sr_fusion50",
    )
    results: dict[str, list[pd.DataFrame]] = {name: [] for name in variants}
    report: dict[str, object] = {
        "protocol": {
            "fit": "normal June 13–21, CatBoost parameters identical across variants",
            "threshold": "previous seven labelled days",
            "SR": "causal 3-day endpoint; SR(t-1) for model feature",
            "fusion_normalization": "QuantileTransformer on preceding seven days' normal scores",
            "fusion_weights": [0.25, 0.50],
        },
        "sensors": {},
    }

    for sensor_id in sensor_ids:
        frame = add_calendar(baseline.make_sensor_frame(values, labels, sensor_id))
        frame = add_sr(frame)
        report["sensors"][sensor_id] = {}
        for model_variant in ("original", "cyclic", "cyclic_scaled", "cyclic_sr_lag1"):
            scored, info = score_model(frame, sensor_id, model_variant, args.max_days)
            evaluated = apply_past_thresholds(scored)
            results[model_variant].append(evaluated)
            report["sensors"][sensor_id][model_variant] = {
                "fit": info,
                "metrics": summarize(evaluated),
            }
            if model_variant == "cyclic":
                for sr_weight in (0.25, 0.50):
                    name = f"cyclic_sr_fusion{int(sr_weight * 100)}"
                    fused = fuse_scores(scored, sr_weight)
                    results[name].append(fused)
                    report["sensors"][sensor_id][name] = {
                        "metrics": summarize(fused)
                    }
        print(f"[{sensor_id}] CatBoost variants complete", flush=True)

    report["pooled"] = {
        name: summarize(pd.concat(model_results).sort_index())
        for name, model_results in results.items()
    }
    print("RESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
