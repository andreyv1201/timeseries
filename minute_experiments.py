from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.seasonal import STL

import catboost_model as cat_base
import evaluation_common as common


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "representation_experiments"
sys.path.insert(0, str(OUTPUT_DIR / ".packages"))
sys.path.insert(0, str(ROOT / ".xgboost_packages"))
import pywt
from ssalib import SingularSpectrumAnalysis
from xgboost import XGBRegressor
import xgboost_model as xgb_base


STEP = pd.Timedelta(minutes=1)
POINTS_PER_DAY = 1440
LAGS = (1440, 2880, 4320)
DECOMP_DAYS = 7
DECOMP_POINTS = DECOMP_DAYS * POINTS_PER_DAY
SSA_EMBEDDING = POINTS_PER_DAY
SSA_RANK = 4
WAVELET = "db4"
WAVELET_LEVEL = 7
WAVELET_WINDOW = POINTS_PER_DAY
CACHE_VERSION = 1
ALARM_POINTS = {"immediate": 1, "30min": 30, "1h": 60, "2h": 120}

BASE_FEATURES = [
    *(f"lag_{lag}" for lag in LAGS),
    "time_sin", "time_cos", "dow_sin", "dow_cos",
]
WAVELET_FEATURES = [
    "wavelet_low_last",
    *(f"wavelet_d{level}_energy" for level in range(1, WAVELET_LEVEL + 1)),
]
STL_FEATURES = ["stl_expected", "stl_trend", "stl_seasonal"]
SSA_FEATURES = ["ssa_expected", "ssa_reconstruction_error"]
MEDIAN_FEATURES = ["seasonal_median_expected"]
EXTRAS = {
    "base": [],
    "wavelet": WAVELET_FEATURES,
    "stl": STL_FEATURES,
    "ssa": SSA_FEATURES,
    "seasonal_median": MEDIAN_FEATURES,
    "all": [*WAVELET_FEATURES, *STL_FEATURES, *SSA_FEATURES],
}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.read_csv(ROOT / "mv_cond_temperature.csv")
    labels = pd.read_csv(ROOT / "labled_mv_cond.csv")
    values["minute"] = pd.to_datetime(
        values["event_timestamp"], unit="s"
    ).dt.floor("min")
    labels["minute"] = pd.to_datetime(
        labels["event_timestamp"], unit="s"
    ).dt.floor("min")
    return values, labels


def make_minute_frame(
    values: pd.DataFrame, labels: pd.DataFrame, sensor_id: str
) -> pd.DataFrame:
    sensor_values = values.loc[values["io_id"].eq(sensor_id)].sort_values("minute")
    sensor_labels = labels.loc[labels["io_id"].eq(sensor_id)].sort_values("minute")
    if sensor_values["minute"].duplicated().any() or sensor_labels["minute"].duplicated().any():
        raise ValueError(f"Minute collisions for sensor {sensor_id}")
    if not sensor_values[["minute", "event_timestamp"]].reset_index(drop=True).equals(
        sensor_labels[["minute", "event_timestamp"]].reset_index(drop=True)
    ):
        raise ValueError(f"Temperature and label timestamps differ for {sensor_id}")

    index = pd.date_range(
        sensor_values["minute"].min(), sensor_values["minute"].max(), freq=STEP
    )
    y = pd.to_numeric(
        sensor_values.set_index("minute")["value"], errors="coerce"
    ).reindex(index)
    label = pd.to_numeric(
        sensor_labels.set_index("minute")["anomaly"], errors="coerce"
    ).reindex(index)
    frame = pd.DataFrame({"y": y, "label": label}, index=index)
    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
        frame[f"label_lag_{lag}"] = frame["label"].shift(lag)
    slot = frame.index.hour * 60 + frame.index.minute
    daily_angle = 2 * np.pi * slot / POINTS_PER_DAY
    weekly_angle = 2 * np.pi * frame.index.dayofweek / 7
    frame["time_sin"] = np.sin(daily_angle)
    frame["time_cos"] = np.cos(daily_angle)
    frame["dow_sin"] = np.sin(weekly_angle)
    frame["dow_cos"] = np.cos(weekly_angle)
    return frame


def add_wavelet_features(frame: pd.DataFrame) -> None:
    filled = frame["y"].ffill().to_numpy(dtype=float)
    columns = {name: np.full(len(frame), np.nan) for name in WAVELET_FEATURES}
    for target in range(WAVELET_WINDOW, len(frame)):
        window = filled[target - WAVELET_WINDOW : target]
        if not np.isfinite(window).all():
            continue
        coefficients = pywt.wavedec(
            window, WAVELET, mode="symmetric", level=WAVELET_LEVEL
        )
        low = pywt.waverec(
            [coefficients[0], *[np.zeros_like(c) for c in coefficients[1:]]],
            WAVELET, mode="symmetric",
        )
        columns["wavelet_low_last"][target] = float(low[-1])
        for level in range(1, WAVELET_LEVEL + 1):
            detail = coefficients[-level]
            columns[f"wavelet_d{level}_energy"][target] = float(
                np.log1p(np.mean(detail * detail))
            )
    for name, data in columns.items():
        frame[name] = data


def ssa_reconstruction(past: np.ndarray) -> np.ndarray:

    ssa = SingularSpectrumAnalysis(
        past, window=SSA_EMBEDDING,
        svd_solver="scipy_sparse", standardize=False,
    ).decompose(
        n_components=SSA_RANK, which="LM", tol=1e-5,
        v0=np.ones(SSA_EMBEDDING, dtype=float),
    )
    return np.asarray(ssa[:SSA_RANK], dtype=float)


def add_decomposition_features(frame: pd.DataFrame, sensor_id: str) -> None:
    filled = frame["y"].ffill()
    columns = {
        name: np.full(len(frame), np.nan)
        for name in [*STL_FEATURES, *SSA_FEATURES, *MEDIAN_FEATURES]
    }
    first_day = frame.index.min().normalize() + pd.Timedelta(days=DECOMP_DAYS + 1)
    last_day = frame.index.max().normalize()
    days = pd.date_range(first_day, last_day, freq="D")
    for number, day in enumerate(days, start=1):
        past_index = pd.date_range(
            day - pd.Timedelta(days=DECOMP_DAYS),
            periods=DECOMP_POINTS, freq=STEP,
        )
        past = filled.reindex(past_index).to_numpy(dtype=float)
        if not np.isfinite(past).all():
            continue
        stl = STL(
            past, period=POINTS_PER_DAY, robust=True,
            seasonal_jump=70, trend_jump=150, low_pass_jump=70,
        ).fit()
        seasonal_template = np.median(
            stl.seasonal[-3 * POINTS_PER_DAY :].reshape(3, POINTS_PER_DAY),
            axis=0,
        )
        expected_stl = float(stl.trend[-1]) + seasonal_template
        expected_median = np.median(
            past.reshape(DECOMP_DAYS, POINTS_PER_DAY), axis=0
        )
        reconstruction = ssa_reconstruction(past)
        expected_ssa = reconstruction[-POINTS_PER_DAY:]
        ssa_error = float(np.mean(np.abs(past[-POINTS_PER_DAY:] - expected_ssa)))
        target_index = pd.date_range(day, periods=POINTS_PER_DAY, freq=STEP)
        positions = frame.index.get_indexer(target_index)
        valid = positions >= 0
        columns["stl_expected"][positions[valid]] = expected_stl[valid]
        columns["stl_trend"][positions[valid]] = stl.trend[-1]
        columns["stl_seasonal"][positions[valid]] = seasonal_template[valid]
        columns["ssa_expected"][positions[valid]] = expected_ssa[valid]
        columns["ssa_reconstruction_error"][positions[valid]] = ssa_error
        columns["seasonal_median_expected"][positions[valid]] = expected_median[valid]
        if number == 1 or number % 10 == 0 or number == len(days):
            print(f"{sensor_id[-2:]} minute decompositions {number}/{len(days)}", flush=True)
    for name, data in columns.items():
        frame[name] = data


def prepare_frame(
    values: pd.DataFrame, labels: pd.DataFrame, sensor_id: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / f"minute_features_v{CACHE_VERSION}_{sensor_id}.pkl"
    if use_cache and cache_path.exists():
        frame = pd.read_pickle(cache_path)
        expected = [*BASE_FEATURES, *WAVELET_FEATURES, *STL_FEATURES,
                    *SSA_FEATURES, *MEDIAN_FEATURES]
        if all(name in frame for name in expected):
            print(f"Loaded minute features for {sensor_id} from cache", flush=True)
            return frame
    frame = make_minute_frame(values, labels, sensor_id)
    print(f"{sensor_id[-2:]} minute wavelet features", flush=True)
    add_wavelet_features(frame)
    add_decomposition_features(frame, sensor_id)
    frame.to_pickle(cache_path)
    print(f"Saved minute feature cache for {sensor_id}", flush=True)
    return frame


def usable_rows(frame: pd.DataFrame) -> pd.Series:
    return frame[["y", "label", *BASE_FEATURES]].notna().all(axis=1)


def normal_training_rows(frame: pd.DataFrame) -> pd.Series:
    normal = usable_rows(frame) & frame["label"].eq(0)
    for lag in LAGS:
        normal &= frame[f"label_lag_{lag}"].eq(0)
    return normal


def selected_rows(
    frame: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = usable_rows(frame) & frame[features].notna().all(axis=1)
    fit_mask = (
        (frame.index >= common.TRAIN_START)
        & (frame.index < common.TRAIN_END)
        & normal_training_rows(frame)
        & frame[features].notna().all(axis=1)
    )
    train = frame.loc[fit_mask]
    score = frame.loc[(frame.index >= common.TRAIN_END) & complete]
    if len(train) < 2 * POINTS_PER_DAY:
        raise ValueError(f"Only {len(train)} minute normal train rows")
    return train, score


def catboost_scores(
    frame: pd.DataFrame, sensor_id: str, variant: str
) -> tuple[pd.DataFrame, dict]:
    features = [*BASE_FEATURES, *EXTRAS[variant]]
    train, score = selected_rows(frame, features)
    model = CatBoostRegressor(**cat_base.CATBOOST_PARAMS)
    model.fit(train[features], train["y"])
    prediction = model.predict(score[features])
    scored = pd.DataFrame({
        "sensor": sensor_id, "label": score["label"].astype(int),
        "score": np.abs(score["y"].to_numpy() - prediction),
    }, index=score.index)
    return scored, {"train_rows": len(train), "features": features}


def scaled_xgb_xy(
    rows: pd.DataFrame, features: list[str], scaler: RobustScaler
) -> tuple[pd.DataFrame, np.ndarray]:
    x = rows[features].copy()
    for lag in LAGS:
        name = f"lag_{lag}"
        x[name] = scaler.transform(x[[name]].to_numpy()).ravel()
    y = scaler.transform(rows[["y"]].to_numpy()).ravel()
    return x, y


def xgboost_scores(
    frame: pd.DataFrame, sensor_id: str, variant: str
) -> tuple[pd.DataFrame, dict]:
    features = [*BASE_FEATURES, *EXTRAS[variant]]
    train, score = selected_rows(frame, features)
    cut = int(len(train) * 0.8)
    inner_train, validation = train.iloc[:cut], train.iloc[cut:]
    inner_scaler = RobustScaler().fit(inner_train[["y"]].to_numpy())
    x_train, y_train = scaled_xgb_xy(inner_train, features, inner_scaler)
    x_valid, y_valid = scaled_xgb_xy(validation, features, inner_scaler)
    selector = XGBRegressor(
        n_estimators=2000, early_stopping_rounds=75,
        **xgb_base.MODEL_PARAMS,
    )
    selector.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    trees = int(selector.best_iteration) + 1
    scaler = RobustScaler().fit(train[["y"]].to_numpy())
    x_all, y_all = scaled_xgb_xy(train, features, scaler)
    model = XGBRegressor(n_estimators=trees, **xgb_base.MODEL_PARAMS)
    model.fit(x_all, y_all, verbose=False)
    x_score, y_score = scaled_xgb_xy(score, features, scaler)
    scored = pd.DataFrame({
        "sensor": sensor_id, "label": score["label"].astype(int),
        "score": np.abs(y_score - model.predict(x_score)),
    }, index=score.index)
    return scored, {
        "train_rows": len(train), "validation_rows": len(validation),
        "selected_trees": trees, "features": features,
        "scaler_center": float(scaler.center_[0]),
        "scaler_scale": float(scaler.scale_[0]),
    }


def decomposition_scores(
    frame: pd.DataFrame, sensor_id: str, variant: str
) -> tuple[pd.DataFrame, dict]:
    expected = f"{variant}_expected"
    mask = (
        (frame.index >= common.TRAIN_END)
        & usable_rows(frame)
        & frame[expected].notna()
    )
    selected = frame.loc[mask]
    scored = pd.DataFrame({
        "sensor": sensor_id, "label": selected["label"].astype(int),
        "score": np.abs(selected["y"] - selected[expected]),
    }, index=selected.index)
    return scored, {"expected_feature": expected, "train_rows": 0}


def confirm_alarm(
    index: pd.DatetimeIndex, raw: np.ndarray, min_points: int
) -> np.ndarray:
    result = np.zeros(len(raw), dtype=np.uint8)
    run_length = 0
    previous_time = None
    for i, (timestamp, prediction) in enumerate(zip(index, raw)):
        if previous_time is None or timestamp - previous_time != STEP:
            run_length = 0
        run_length = run_length + 1 if prediction else 0
        result[i] = int(run_length >= min_points)
        previous_time = timestamp
    return result


def apply_past_thresholds(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.sort_index()
    days = pd.date_range(
        common.FIRST_TEST_DAY, scored.index.max().normalize(), freq="D"
    )
    previous = None
    evaluated = []
    for day in days:
        calibration = scored.loc[
            (scored.index >= day - pd.Timedelta(days=7)) & (scored.index < day)
        ]
        current = scored.loc[
            (scored.index >= day) & (scored.index < day + pd.Timedelta(days=1))
        ].copy()
        if calibration.empty or current.empty:
            continue
        threshold, source = common.choose_threshold(
            calibration["label"].to_numpy(), calibration["score"].to_numpy(), previous
        )
        previous = threshold
        current["threshold"] = threshold
        current["threshold_source"] = source
        current["raw_prediction"] = (current["score"] >= threshold).astype(np.uint8)
        evaluated.append(current)
    if not evaluated:
        raise ValueError("No minute test days")
    result = pd.concat(evaluated).sort_index()
    for name, points in ALARM_POINTS.items():
        result[f"alarm_{name}"] = confirm_alarm(
            result.index, result["raw_prediction"].to_numpy(), points
        )
    return result


def summarize(result: pd.DataFrame) -> dict:
    return {
        name: common.metric_values(result, f"alarm_{name}")
        for name in ALARM_POINTS
    }


def run(sensor_filter: str | None, max_days: int | None) -> dict:
    values, labels = load_raw()
    sensor_ids = [str(s) for s in values["io_id"].dropna().unique()]
    if sensor_filter is not None:
        if sensor_filter not in sensor_ids:
            raise ValueError(f"Unknown sensor: {sensor_filter}")
        sensor_ids = [sensor_filter]
    variants = [
        *(f"CatBoost+{name}" for name in EXTRAS),
        *(f"XGBoost+{name}" for name in EXTRAS),
        "STL residual", "SSA residual",
    ]
    report = {
        "protocol": {
            "source": "original one-minute records; timestamps floored to minute, no averaging",
            "minute_grid": "missing minutes remain unobserved targets; past-only ffill for transforms",
            "train": "normal June 13-21; normal target and daily-lag labels",
            "model_validation": "XGBoost last 20% normal train, chronological",
            "test_start": "2025-06-29",
            "threshold": "preceding seven labelled days per sensor",
            "alarm": "60 consecutive scored one-minute points for 1h",
            "wavelet": "db4 level 7 from previous 1440 one-minute points",
            "stl": "previous seven days, period 1440, robust, daily update",
            "ssa": "previous seven days, embedding 1440, rank 4, daily update",
            "max_days": max_days,
        },
        "sensors": {}, "pooled": {},
    }
    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in variants}
    for sensor_id in sensor_ids:
        print(f"Preparing one-minute data for {sensor_id}", flush=True)
        frame = prepare_frame(values, labels, sensor_id)
        sensor_results: dict[str, dict] = {}
        reference_index = None
        for name in variants:
            if name.startswith("CatBoost+"):
                scored, info = catboost_scores(frame, sensor_id, name.split("+", 1)[1])
            elif name.startswith("XGBoost+"):
                scored, info = xgboost_scores(frame, sensor_id, name.split("+", 1)[1])
            else:
                scored, info = decomposition_scores(
                    frame, sensor_id, name.split(" ")[0].lower()
                )
            if max_days is not None:
                scored = scored.loc[
                    scored.index < common.FIRST_TEST_DAY + pd.Timedelta(days=max_days)
                ]
            evaluated = apply_past_thresholds(scored)
            if reference_index is None:
                reference_index = evaluated.index
            elif not evaluated.index.equals(reference_index):
                raise AssertionError(f"Mismatched one-minute test rows: {sensor_id} {name}")
            metrics = summarize(evaluated)
            sensor_results[name] = {
                "fit": info,
                "first_test_point": str(evaluated.index.min()),
                "last_test_point": str(evaluated.index.max()),
                "metrics": metrics,
            }
            pooled[name].append(evaluated)
            one_hour = metrics["1h"]
            roc = one_hour["roc_auc"]
            ap = one_hour["average_precision"]
            print(
                f"{sensor_id[-2:]} {name}: n={len(evaluated)} "
                f"ROC={roc if roc is not None else 'n/a'} "
                f"AP={ap if ap is not None else 'n/a'} "
                f"F1@1h={one_hour['f1']:.3f}", flush=True,
            )
        report["sensors"][sensor_id] = sensor_results
    for name, datasets in pooled.items():
        report["pooled"][name] = summarize(pd.concat(datasets).sort_index())
    OUTPUT_DIR.mkdir(exist_ok=True)
    name = "minute_classical_smoke.json" if max_days is not None else "minute_classical_results.json"
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    print("RESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()
    run(args.sensor, args.max_days)
