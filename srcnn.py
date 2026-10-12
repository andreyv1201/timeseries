from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_CACHE_DISABLE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, TensorDataset
from evaluation_common import confirm_alarm


DATA_PATH = "mv_cond_temperature.csv"
LABEL_PATH = "labled_mv_cond.csv"
RESAMPLE_RULE = "10min"

POINTS_PER_DAY = 144
ONE_HOUR_ALARM_POINTS = 6
FIT_DAYS = 21
CALIBRATION_DAYS = 7
DEVELOPMENT_DAYS = 21
FORECAST_DAYS = 1
STEP_DAYS = 1

WINDOW = 3 * POINTS_PER_DAY
TRAIN_STRIDE = 3
BATCH_SIZE = 256
EPOCHS = 10
CHANNELS = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_TRAIN_WINDOWS = 30
POOLING = "endpoint"
SPECTRAL_SCORE_WEIGHT = 4.0
NORMAL_ONLY_QUANTILE = 0.99
SEED = 42

ESTIMATED_POINTS = 5
GRADIENT_LOOKBACK = 5
AMP_WINDOW = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ThresholdChoice:
    value: float
    source: str


@dataclass(frozen=True)
class FeatureCache:
    real: np.ndarray


class SRAutoencoder(nn.Module):


    def __init__(
        self, channels: int = CHANNELS, pooling: str = POOLING
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(
                channels * 2,
                channels * 2,
                kernel_size=27,
                stride=27,
            ),
            nn.ReLU(),
        )
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2 * 16, 32),
            nn.ReLU(),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(32, channels * 2 * 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        latent = self.to_latent(encoded)
        decoded = self.from_latent(latent).view(
            x.shape[0], CHANNELS * 2, 16
        )
        decoded = F.interpolate(decoded, size=x.shape[-1], mode="nearest")
        return self.decoder(decoded)

    def anomaly_score(
        self, x: torch.Tensor, reconstruction: torch.Tensor
    ) -> torch.Tensor:
        squared_error = (x - reconstruction).square()
        if self.pooling == "global":
            reconstruction_score = squared_error[:, :, :WINDOW].mean(dim=(1, 2))
        elif self.pooling == "endpoint":
            reconstruction_score = squared_error[:, :, WINDOW - 1].mean(dim=1)
        elif self.pooling == "tail6":
            reconstruction_score = squared_error[
                :, :, WINDOW - 6 : WINDOW
            ].mean(dim=(1, 2))
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")
        spectral_score = torch.relu(x[:, :, WINDOW - 1]).square().mean(dim=1)
        return reconstruction_score + SPECTRAL_SCORE_WEIGHT * spectral_score


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


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
    return pd.DataFrame(
        {
            "value": value_series,
            "label": label_series,
            "observed": value_series.notna(),
        }
    )


def moving_average_frequency(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x.copy()
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(x, kernel, mode="same")


def extrapolated_tail(window: np.ndarray) -> np.ndarray:

    lookback = min(GRADIENT_LOOKBACK, len(window) - 1)
    if lookback <= 0:
        next_value = float(window[-1])
    else:
        gradients = [
            (float(window[-1]) - float(window[-1 - lag])) / lag
            for lag in range(1, lookback + 1)
        ]
        next_value = float(window[-lookback]) + float(np.mean(gradients)) * lookback
    return np.full(ESTIMATED_POINTS, next_value, dtype=np.float64)


def spectral_residual_feature(window: np.ndarray) -> np.ndarray:

    values = np.asarray(window, dtype=np.float64)
    extended = np.concatenate([values, extrapolated_tail(values)])
    spectrum = np.fft.fft(extended)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    log_magnitude = np.log(magnitude + 1e-8)
    residual = log_magnitude - moving_average_frequency(log_magnitude, AMP_WINDOW)
    saliency = np.abs(np.fft.ifft(np.exp(residual + 1j * phase)))
    median = float(np.median(saliency))
    mad = float(np.median(np.abs(saliency - median)))
    scale = max(1.4826 * mad, 1e-6)
    return ((saliency - median) / scale).astype(np.float32)


def precompute_feature_cache(frame: pd.DataFrame) -> FeatureCache:

    values = frame["value"].ffill().to_numpy(dtype=np.float64)
    feature_length = WINDOW + ESTIMATED_POINTS
    real = np.full((len(values), feature_length), np.nan, dtype=np.float32)
    for endpoint in range(WINDOW - 1, len(values)):
        raw = values[endpoint - WINDOW + 1 : endpoint + 1]
        if not np.isfinite(raw).all():
            continue
        real[endpoint] = spectral_residual_feature(raw)
    return FeatureCache(real=real)


def make_loader(features: np.ndarray, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def fit_daily_model(
    frame: pd.DataFrame,
    cache: FeatureCache,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    pooling: str,
) -> tuple[SRAutoencoder | None, int]:
    fit_index = pd.date_range(
        fit_start,
        fit_end - pd.Timedelta(RESAMPLE_RULE),
        freq=RESAMPLE_RULE,
    )
    fit_frame = frame.reindex(fit_index)
    usable = (
        fit_frame["observed"].fillna(False).to_numpy(dtype=bool)
        & fit_frame["label"].eq(0).to_numpy(dtype=bool)
    )
    normal_windows = np.lib.stride_tricks.sliding_window_view(
        usable, WINDOW
    ).all(axis=1)
    local_endpoints = np.flatnonzero(normal_windows) + WINDOW - 1
    local_endpoints = local_endpoints[::TRAIN_STRIDE]
    if len(local_endpoints) < MIN_TRAIN_WINDOWS:
        return None, len(local_endpoints)

    frame_positions = frame.index.get_indexer(fit_index)
    if np.any(frame_positions < 0):
        raise ValueError("Training interval is outside the sensor frame")
    endpoints = frame_positions[local_endpoints]

    features = cache.real[endpoints, None, :]
    seed_everything(SEED)
    model = SRAutoencoder(pooling=pooling).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()
    loader = make_loader(features, True, SEED)

    for _ in range(EPOCHS):
        model.train()
        for (batch_features,) in loader:
            batch_features = batch_features.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_features)
            loss.backward()
            optimizer.step()

    return model, len(endpoints)


def score_period(
    model: SRAutoencoder,
    frame: pd.DataFrame,
    cache: FeatureCache,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
) -> pd.DataFrame:
    step = pd.Timedelta(RESAMPLE_RULE)
    target_index = pd.date_range(score_start, score_end - step, freq=RESAMPLE_RULE)
    actual = frame["value"].reindex(target_index)
    labels = frame["label"].reindex(target_index)
    positions = frame.index.get_indexer(target_index)
    if np.any(positions < 0):
        raise ValueError("Scoring interval is outside the sensor frame")
    features = cache.real[positions, None, :]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(DEVICE)
            reconstruction = model(batch_features)
            probabilities.append(
                model.anomaly_score(batch_features, reconstruction).cpu().numpy()
            )
    scores = np.concatenate(probabilities).astype(np.float64)
    if len(scores) != len(target_index):
        raise AssertionError("One SR-CNN score per target point was expected")

    valid = actual.notna() & labels.notna() & np.isfinite(scores)
    valid_array = np.asarray(valid, dtype=bool)
    return pd.DataFrame(
        {
            "label": labels.loc[valid].to_numpy(dtype=int),
            "score": scores[valid_array],
        },
        index=target_index[valid],
    )


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    previous_threshold: float | None,
) -> ThresholdChoice:
    valid = np.isfinite(y_true) & np.isfinite(scores)
    y_true = y_true[valid].astype(int)
    scores = scores[valid].astype(float)
    classes = np.unique(y_true)

    if len(scores) == 0:
        if previous_threshold is None:
            raise ValueError("No preceding labelled calibration scores")
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


def daily_walk_forward(
    frame: pd.DataFrame,
    cache: FeatureCache,
    sensor_id: str,
    max_days: int | None,
    pooling: str,
) -> pd.DataFrame:
    first_complete_day = frame.index.min().normalize() + pd.Timedelta(days=1)
    first_forecast_day = first_complete_day + pd.Timedelta(
        days=FIT_DAYS + CALIBRATION_DAYS
    )
    final_forecast_day = frame.index.max().normalize()
    forecast_days = pd.date_range(
        first_forecast_day, final_forecast_day, freq=f"{STEP_DAYS}D"
    )
    if max_days is not None:
        forecast_days = forecast_days[:max_days]

    previous_threshold: float | None = None
    model: SRAutoencoder | None = None
    daily_results: list[pd.DataFrame] = []

    for number, forecast_start in enumerate(forecast_days, start=1):
        calibration_start = forecast_start - pd.Timedelta(days=CALIBRATION_DAYS)
        fit_start = calibration_start - pd.Timedelta(days=FIT_DAYS)
        new_model, normal_windows = fit_daily_model(
            frame, cache, fit_start, calibration_start, pooling
        )
        if new_model is None:
            if model is None:
                raise ValueError(
                    f"[{sensor_id}] no earlier model is available and only "
                    f"{normal_windows} normal training windows were found"
                )
            model_source = "reused_previous"
        else:
            if model is not None:
                del model
            model = new_model
            model_source = "trained"

        scored = score_period(
            model,
            frame,
            cache,
            calibration_start,
            forecast_start + pd.Timedelta(days=FORECAST_DAYS),
        )
        calibration = scored.loc[scored.index < forecast_start]
        current = scored.loc[scored.index >= forecast_start].copy()
        threshold = choose_threshold(
            calibration["label"].to_numpy(dtype=float),
            calibration["score"].to_numpy(dtype=float),
            previous_threshold,
        )
        previous_threshold = threshold.value
        current["prediction"] = (
            current["score"].to_numpy(dtype=float) >= threshold.value
        ).astype(int)
        current["threshold"] = threshold.value
        current["threshold_source"] = threshold.source
        current["forecast_day"] = forecast_start
        current["sensor"] = sensor_id
        current["model_source"] = model_source
        daily_results.append(current)

        if number == 1 or number % 10 == 0 or number == len(forecast_days):
            print(
                f"[{sensor_id}] forecast {number}/{len(forecast_days)} "
                f"day={forecast_start.date()} train_windows={normal_windows} "
                f"model={model_source} threshold={threshold.value:.6f}",
                flush=True,
            )

    if not daily_results:
        raise ValueError(f"[{sensor_id}] no SR-CNN forecast days")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat(daily_results).sort_index()


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


def print_metrics(name: str, metrics: dict[str, object]) -> None:
    print(f"\n{name}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def self_test() -> None:
    seed_everything()
    window = np.linspace(2.0, 7.0, WINDOW)
    feature = spectral_residual_feature(window)
    if len(feature) != WINDOW + ESTIMATED_POINTS or not np.isfinite(feature).all():
        raise AssertionError("Invalid spectral-residual feature")
    model = SRAutoencoder()
    batch = torch.from_numpy(np.stack([feature, feature])[:, None, :])
    reconstruction = model(batch)
    if reconstruction.shape != batch.shape:
        raise AssertionError("Unexpected SR autoencoder output shape")
    if model.anomaly_score(batch, reconstruction).shape != (2,):
        raise AssertionError("Unexpected SR autoencoder score shape")
    print("SR feature, autoencoder reconstruction, and score checks OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--pooling", choices=("global", "endpoint", "tail6"), default=POOLING
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    seed_everything()
    values, labels = load_panel()
    sensor_ids = [str(value) for value in values["io_id"].dropna().unique()]
    if args.sensor is not None:
        if args.sensor not in sensor_ids:
            raise ValueError(f"Unknown sensor: {args.sensor}")
        sensor_ids = [args.sensor]

    print(f"device: {DEVICE}")
    all_results: list[pd.DataFrame] = []
    all_late_results: list[pd.DataFrame] = []
    report: dict[str, object] = {
        "protocol": {
            "fit_days": FIT_DAYS,
            "calibration_days": CALIBRATION_DAYS,
            "development_days_for_pooling": DEVELOPMENT_DAYS,
            "forecast_days": FORECAST_DAYS,
            "step_days": STEP_DAYS,
            "window_points": WINDOW,
            "window_days": WINDOW / POINTS_PER_DAY,
            "train_stride_points": TRAIN_STRIDE,
            "epochs": EPOCHS,
            "minimum_training_windows": MIN_TRAIN_WINDOWS,
            "insufficient_training_fallback": "reuse preceding fitted model",
            "threshold": "preceding seven labelled days rescored by current model",
            "score_alignment": "window endpoint; causal",
            "training_windows": "past fully observed normal windows only",
            "training_objective": "reconstruct normal spectral-residual windows",
            "synthetic_anomaly": "not used",
            "device": str(DEVICE),
            "seed": SEED,
            "pooling": args.pooling,
            "spectral_score_weight": SPECTRAL_SCORE_WEIGHT,
            "one_hour_alarm_points": ONE_HOUR_ALARM_POINTS,
        },
        "sensors": {},
    }

    with threadpool_limits(limits=1):
        for sensor_id in sensor_ids:
            frame = make_sensor_frame(values, labels, sensor_id)
            print(f"[{sensor_id}] precomputing causal SR features", flush=True)
            cache = precompute_feature_cache(frame)
            result = daily_walk_forward(
                frame, cache, sensor_id, args.max_days, args.pooling
            )
            result["causal_1h_prediction"] = confirm_alarm(
                result.index,
                result["prediction"].to_numpy(dtype=int),
                ONE_HOUR_ALARM_POINTS,
            )
            all_results.append(result)
            metrics = metric_values(
                result["label"], result["score"], result["prediction"]
            )
            one_hour_metrics = metric_values(
                result["label"], result["score"], result["causal_1h_prediction"]
            )
            result_days = pd.DatetimeIndex(
                result["forecast_day"].drop_duplicates()
            ).sort_values()
            late_days = result_days[DEVELOPMENT_DAYS:]
            late_result = result.loc[result["forecast_day"].isin(late_days)]
            late_metrics = (
                metric_values(
                    late_result["label"],
                    late_result["score"],
                    late_result["prediction"],
                )
                if len(late_result)
                else None
            )
            late_one_hour_metrics = (
                metric_values(
                    late_result["label"],
                    late_result["score"],
                    late_result["causal_1h_prediction"],
                )
                if len(late_result)
                else None
            )
            if len(late_result):
                all_late_results.append(late_result)
            report["sensors"][sensor_id] = {
                "first_prediction": result.index.min().isoformat(),
                "last_prediction": result.index.max().isoformat(),
                "reused_model_days": int(
                    result.loc[
                        result["model_source"].eq("reused_previous"),
                        "forecast_day",
                    ].nunique()
                ),
                "metrics": metrics,
                "causal_1h_metrics": one_hour_metrics,
                "late_holdout_start": (
                    late_result.index.min().isoformat() if len(late_result) else None
                ),
                "late_holdout_metrics": late_metrics,
                "late_holdout_causal_1h_metrics": late_one_hour_metrics,
            }
            print_metrics(f"SENSOR {sensor_id}", metrics)
            print_metrics(f"SENSOR {sensor_id} — causal 1h alarm", one_hour_metrics)
            if late_metrics is not None:
                print_metrics(f"SENSOR {sensor_id} LATE HOLDOUT", late_metrics)
                print_metrics(
                    f"SENSOR {sensor_id} LATE HOLDOUT — causal 1h alarm",
                    late_one_hour_metrics,
                )

    pooled = pd.concat(all_results).sort_index()
    pooled_metrics = metric_values(
        pooled["label"], pooled["score"], pooled["prediction"]
    )
    pooled_one_hour_metrics = metric_values(
        pooled["label"], pooled["score"], pooled["causal_1h_prediction"]
    )
    sensor_metrics = [entry["metrics"] for entry in report["sensors"].values()]
    sensor_one_hour_metrics = [
        entry["causal_1h_metrics"] for entry in report["sensors"].values()
    ]
    report["macro_metrics"] = macro_metrics(sensor_metrics)
    report["pooled_metrics"] = pooled_metrics
    report["macro_causal_1h_metrics"] = macro_metrics(sensor_one_hour_metrics)
    report["pooled_causal_1h_metrics"] = pooled_one_hour_metrics
    print_metrics("POOLED", pooled_metrics)
    print_metrics("POOLED — causal 1h alarm", pooled_one_hour_metrics)
    if all_late_results:
        pooled_late = pd.concat(all_late_results).sort_index()
        pooled_late_metrics = metric_values(
            pooled_late["label"],
            pooled_late["score"],
            pooled_late["prediction"],
        )
        pooled_late_one_hour_metrics = metric_values(
            pooled_late["label"],
            pooled_late["score"],
            pooled_late["causal_1h_prediction"],
        )
        late_sensor_metrics = [
            entry["late_holdout_metrics"]
            for entry in report["sensors"].values()
            if entry["late_holdout_metrics"] is not None
        ]
        report["late_holdout_macro_metrics"] = macro_metrics(late_sensor_metrics)
        report["late_holdout_pooled_metrics"] = pooled_late_metrics
        late_sensor_one_hour_metrics = [
            entry["late_holdout_causal_1h_metrics"]
            for entry in report["sensors"].values()
            if entry["late_holdout_causal_1h_metrics"] is not None
        ]
        report["late_holdout_macro_causal_1h_metrics"] = macro_metrics(
            late_sensor_one_hour_metrics
        )
        report["late_holdout_pooled_causal_1h_metrics"] = (
            pooled_late_one_hour_metrics
        )
        print_metrics("LATE HOLDOUT POOLED", pooled_late_metrics)
        print_metrics(
            "LATE HOLDOUT POOLED — causal 1h alarm",
            pooled_late_one_hour_metrics,
        )
    print("\nRESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
