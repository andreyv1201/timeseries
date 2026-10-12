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
sys.path.insert(0, str(ROOT / "representation_experiments" / ".packages"))
sys.path.insert(0, str(ROOT / ".xgboost_packages"))
import pywt
from ssalib import SingularSpectrumAnalysis
from xgboost import XGBRegressor
import xgboost_model as xgb_base


OUTPUT_DIR = ROOT / "representation_experiments"
WAVELET = "db4"
WAVELET_LEVEL = 4
WAVELET_WINDOW = common.POINTS_PER_DAY
DECOMP_DAYS = 7
DECOMP_POINTS = DECOMP_DAYS * common.POINTS_PER_DAY
SSA_EMBEDDING = common.POINTS_PER_DAY
SSA_RANK = 4

WAVELET_FEATURES = [
    "wavelet_low_last",
    "wavelet_d1_energy",
    "wavelet_d2_energy",
    "wavelet_d3_energy",
    "wavelet_d4_energy",
]
STL_FEATURES = ["stl_expected", "stl_trend", "stl_seasonal"]
SSA_FEATURES = ["ssa_expected", "ssa_reconstruction_error"]
SEASONAL_MEDIAN_FEATURES = ["seasonal_median_expected"]
BASE_FEATURES = common.feature_columns()
EXTRAS = {
    "base": [],
    "wavelet": WAVELET_FEATURES,
    "stl": STL_FEATURES,
    "ssa": SSA_FEATURES,
    "seasonal_median": SEASONAL_MEDIAN_FEATURES,
    "all": [*WAVELET_FEATURES, *STL_FEATURES, *SSA_FEATURES],
}


def add_wavelet_features(frame: pd.DataFrame) -> None:

    filled = frame["y"].ffill().to_numpy(dtype=float)
    columns = {name: np.full(len(frame), np.nan) for name in WAVELET_FEATURES}
    zero_details: list[np.ndarray]
    for target in range(WAVELET_WINDOW, len(frame)):
        window = filled[target - WAVELET_WINDOW : target]
        if not np.isfinite(window).all():
            continue
        coefficients = pywt.wavedec(
            window, WAVELET, mode="symmetric", level=WAVELET_LEVEL
        )
        zero_details = [np.zeros_like(part) for part in coefficients[1:]]
        approximation = pywt.waverec(
            [coefficients[0], *zero_details], WAVELET, mode="symmetric"
        )
        columns["wavelet_low_last"][target] = float(approximation[-1])
        for level in range(1, WAVELET_LEVEL + 1):
            detail = coefficients[-level]
            columns[f"wavelet_d{level}_energy"][target] = float(
                np.log1p(np.mean(detail * detail))
            )
    for name, values in columns.items():
        frame[name] = values


def ssa_reconstruction(past: np.ndarray) -> np.ndarray:

    ssa = SingularSpectrumAnalysis(
        past, window=SSA_EMBEDDING,
        svd_solver="scipy_sparse", standardize=False,
    ).decompose(
        n_components=SSA_RANK, which="LM", tol=1e-5,
        v0=np.ones(SSA_EMBEDDING, dtype=float),
    )
    return np.asarray(ssa[:SSA_RANK], dtype=float)


def add_decomposition_features(frame: pd.DataFrame) -> None:

    filled = frame["y"].ffill()
    columns = {
        name: np.full(len(frame), np.nan)
        for name in [*STL_FEATURES, *SSA_FEATURES, *SEASONAL_MEDIAN_FEATURES]
    }
    first_day = frame.index.min().normalize() + pd.Timedelta(days=DECOMP_DAYS + 1)
    last_day = frame.index.max().normalize()
    for day in pd.date_range(first_day, last_day, freq="D"):
        past_index = pd.date_range(
            day - pd.Timedelta(days=DECOMP_DAYS),
            periods=DECOMP_POINTS,
            freq=common.STEP,
        )
        past = filled.reindex(past_index).to_numpy(dtype=float)
        if not np.isfinite(past).all():
            continue
        stl = STL(
            past, period=common.POINTS_PER_DAY, robust=True,
            seasonal_jump=7, trend_jump=15, low_pass_jump=7,
        ).fit()
        seasonal_template = np.median(
            stl.seasonal[-3 * common.POINTS_PER_DAY :].reshape(
                3, common.POINTS_PER_DAY
            ), axis=0,
        )
        expected_stl = float(stl.trend[-1]) + seasonal_template
        expected_median = np.median(
            past.reshape(DECOMP_DAYS, common.POINTS_PER_DAY), axis=0
        )
        ssa_series = ssa_reconstruction(past)
        expected_ssa = ssa_series[-common.POINTS_PER_DAY :]
        ssa_error = float(np.mean(np.abs(
            past[-common.POINTS_PER_DAY :] - expected_ssa
        )))

        target_index = pd.date_range(
            day, periods=common.POINTS_PER_DAY, freq=common.STEP
        )
        positions = frame.index.get_indexer(target_index)
        valid = positions >= 0
        if not valid.any():
            continue
        columns["stl_expected"][positions[valid]] = expected_stl[valid]
        columns["stl_trend"][positions[valid]] = stl.trend[-1]
        columns["stl_seasonal"][positions[valid]] = seasonal_template[valid]
        columns["ssa_expected"][positions[valid]] = expected_ssa[valid]
        columns["ssa_reconstruction_error"][positions[valid]] = ssa_error
        columns["seasonal_median_expected"][positions[valid]] = expected_median[valid]

    for name, values in columns.items():
        frame[name] = values


def prepare_frame(
    values: pd.DataFrame, labels: pd.DataFrame, sensor_id: str
) -> pd.DataFrame:
    frame = common.make_sensor_frame(values, labels, sensor_id)
    add_wavelet_features(frame)
    add_decomposition_features(frame)
    return frame


def selected_rows(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = common.usable_rows(frame) & frame[features].notna().all(axis=1)
    normal = (
        (frame.index >= common.TRAIN_START)
        & (frame.index < common.TRAIN_END)
        & common.normal_training_rows(frame)
        & frame[features].notna().all(axis=1)
    )
    train = frame.loc[normal]
    score = frame.loc[(frame.index >= common.TRAIN_END) & complete]
    if len(train) < 2 * common.POINTS_PER_DAY:
        raise ValueError(f"Only {len(train)} normal training rows")
    return train, score


def catboost_scores(
    frame: pd.DataFrame, sensor_id: str, variant: str
) -> tuple[pd.DataFrame, dict]:
    features = [*BASE_FEATURES, *EXTRAS[variant]]
    train, score = selected_rows(frame, features)
    model = CatBoostRegressor(**cat_base.CATBOOST_PARAMS)
    model.fit(train[features], train["y"])
    residual = np.abs(score["y"].to_numpy() - model.predict(score[features]))
    scored = pd.DataFrame({
        "sensor": sensor_id,
        "label": score["label"].astype(int),
        "score": residual,
    }, index=score.index)
    return scored, {"train_rows": len(train), "features": features}


def scaled_xgb_xy(
    rows: pd.DataFrame, features: list[str], scaler: RobustScaler
) -> tuple[pd.DataFrame, np.ndarray]:
    x = rows[features].copy()
    for lag in common.LAGS:
        name = f"lag_{lag}"
        x[name] = scaler.transform(x[[name]].to_numpy()).ravel()
    y = scaler.transform(rows[["y"]].to_numpy()).ravel()
    return x, y


def xgboost_scores(
    frame: pd.DataFrame, sensor_id: str, variant: str
) -> tuple[pd.DataFrame, dict]:
    features = [*BASE_FEATURES, *EXTRAS[variant]]
    train, score = selected_rows(frame, features)
    split = int(len(train) * 0.8)
    inner_train, validation = train.iloc[:split], train.iloc[split:]
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
    residual = np.abs(y_score - model.predict(x_score))
    scored = pd.DataFrame({
        "sensor": sensor_id,
        "label": score["label"].astype(int),
        "score": residual,
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
        & common.usable_rows(frame)
        & frame[expected].notna()
    )
    selected = frame.loc[mask]
    scored = pd.DataFrame({
        "sensor": sensor_id,
        "label": selected["label"].astype(int),
        "score": np.abs(selected["y"] - selected[expected]),
    }, index=selected.index)
    return scored, {"expected_feature": expected, "train_rows": 0}


def run(sensor_filter: str | None, max_days: int | None) -> dict:
    values, labels = common.load_panel()
    sensor_ids = [str(sensor) for sensor in values["io_id"].dropna().unique()]
    if sensor_filter is not None:
        if sensor_filter not in sensor_ids:
            raise ValueError(f"Unknown sensor {sensor_filter}")
        sensor_ids = [sensor_filter]

    variants = [
        *(f"CatBoost+{name}" for name in EXTRAS),
        *(f"XGBoost+{name}" for name in EXTRAS),
        "STL residual", "SSA residual",
    ]
    result = {
        "protocol": {
            "train": "2025-06-13 through 2025-06-21, normal targets/lag context only",
            "model_validation": "XGBoost last 20% normal train, chronological",
            "test_start": "2025-06-29",
            "threshold": "preceding seven labelled days, separately per sensor",
            "reported_alarm": "immediate and causal 1h (6 successive 10-min points)",
            "wavelet": "db4 level 4 energies and low-pass endpoint from y[t-144:t]",
            "stl": "previous seven complete days, robust daily period 144; forecast trend+median seasonal profile",
            "ssa": "previous seven complete days, embedding 144, rank 4; replay last reconstructed day",
            "seasonal_median_control": "same seven previous days; median of the seven values at each time of day",
            "max_days": max_days,
        },
        "sensors": {}, "pooled": {},
    }
    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in variants}
    for sensor_id in sensor_ids:
        print(f"Preparing causal representations for {sensor_id}", flush=True)
        frame = prepare_frame(values, labels, sensor_id)
        sensor_results: dict[str, dict] = {}
        reference_index: pd.DatetimeIndex | None = None
        for name in variants:
            if name.startswith("CatBoost+"):
                scored, fit_info = catboost_scores(frame, sensor_id, name.split("+", 1)[1])
            elif name.startswith("XGBoost+"):
                scored, fit_info = xgboost_scores(frame, sensor_id, name.split("+", 1)[1])
            else:
                scored, fit_info = decomposition_scores(frame, sensor_id, name.split(" ")[0].lower())
            if max_days is not None:
                scored = scored.loc[
                    scored.index < common.FIRST_TEST_DAY + pd.Timedelta(days=max_days)
                ]
            evaluated = common.apply_past_thresholds(scored)
            if reference_index is None:
                reference_index = evaluated.index
            elif not evaluated.index.equals(reference_index):
                raise AssertionError(f"Mismatched test rows for {sensor_id}: {name}")
            metrics = common.summarize(evaluated)
            sensor_results[name] = {
                "fit": fit_info,
                "first_test_point": str(evaluated.index.min()),
                "last_test_point": str(evaluated.index.max()),
                "metrics": metrics,
            }
            pooled[name].append(evaluated)
            one_hour = metrics["1h"]
            roc_text = (
                f"{one_hour['roc_auc']:.3f}"
                if one_hour["roc_auc"] is not None else "n/a"
            )
            ap_text = (
                f"{one_hour['average_precision']:.3f}"
                if one_hour["average_precision"] is not None else "n/a"
            )
            print(
                f"{sensor_id[-2:]} {name}: n={len(evaluated)} "
                f"ROC={roc_text} AP={ap_text} "
                f"F1@1h={one_hour['f1']:.3f}", flush=True,
            )
        result["sensors"][sensor_id] = sensor_results

    for name, datasets in pooled.items():
        result["pooled"][name] = common.summarize(pd.concat(datasets).sort_index())
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = "classical_smoke.json" if max_days is not None else "classical_results.json"
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()
    run(args.sensor, args.max_days)
