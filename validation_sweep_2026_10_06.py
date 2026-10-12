from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import evaluation_common as common
import representation_experiments as rep


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiment_rerun_2026-10-06"
DEVELOPMENT_START = pd.Timestamp("2025-06-29")
EVALUATION_START = pd.Timestamp("2025-07-20")
DEVELOPMENT_END = EVALUATION_START - pd.Timedelta(seconds=1)
WINDOWS = {
    "wavelet": (144, 288, 432, 720, 1008),
    "stl": (3, 5, 7),
    "ssa": (3, 5, 7),
}
MODELS = {
    "CatBoost": rep.catboost_scores,
    "XGBoost": rep.xgboost_scores,
}


def frame_with_feature(base: pd.DataFrame, kind: str, window: int) -> pd.DataFrame:
    frame = base.copy()
    if kind == "wavelet":
        rep.WAVELET_WINDOW = window
        rep.add_wavelet_features(frame)
    else:
        rep.DECOMP_DAYS = window
        rep.DECOMP_POINTS = window * common.POINTS_PER_DAY
        rep.add_decomposition_features(frame)
    return frame


def rank_metric(scored: pd.DataFrame) -> dict[str, float | int]:
    section = scored.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    y = section["label"].to_numpy(dtype=int)
    s = section["score"].to_numpy(dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("Development interval must contain both classes")
    return {
        "points": int(len(section)),
        "anomalies": int(y.sum()),
        "ap": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }


def history_context_counts(
    frame: pd.DataFrame, window: int, train: pd.DataFrame
) -> dict[str, int]:
    previous_labels = frame["label"].shift(1)
    anomaly = previous_labels.fillna(0).rolling(window, min_periods=window).max().gt(0)
    unknown = previous_labels.isna().astype(int).rolling(window, min_periods=window).sum().gt(0)
    return {
        "normal_target_rows_with_anomaly_in_wavelet_history": int(
            anomaly.reindex(train.index).fillna(False).sum()
        ),
        "normal_target_rows_with_unknown_wavelet_history": int(
            unknown.reindex(train.index).fillna(False).sum()
        ),
    }


def main() -> None:
    values, labels = common.load_panel()
    sensors = [str(s) for s in values["io_id"].dropna().unique()]
    bases = {
        s: common.make_sensor_frame(values, labels, s)
        for s in sensors
    }
    runs: dict[str, dict] = {}
    for kind, candidates in WINDOWS.items():
        for window in candidates:
            key = f"{kind}:{window}"
            runs[key] = {}
            for sensor in sensors:

                base = bases[sensor].loc[:DEVELOPMENT_END]
                frame = frame_with_feature(base, kind, window)
                context = {}
                if kind == "wavelet":
                    train, _ = rep.selected_rows(
                        frame, [*rep.BASE_FEATURES, *rep.WAVELET_FEATURES]
                    )
                    context = history_context_counts(frame, window, train)
                for model, scorer in MODELS.items():
                    scored, fit = scorer(frame, sensor, kind)
                    runs[key].setdefault(model, {})[sensor] = {
                        **rank_metric(scored),
                        "train_rows": fit["train_rows"],
                        **context,
                    }
            for model in MODELS:
                aps = [runs[key][model][s]["ap"] for s in sensors]
                print(f"DEV {model}+{kind}, window={window}: macro AP={np.mean(aps):.5f}", flush=True)

    selected: dict[str, dict[str, int]] = {}
    for model in MODELS:
        selected[model] = {}
        for kind, candidates in WINDOWS.items():
            selected[model][kind] = max(
                candidates,
                key=lambda window: np.mean([
                    runs[f"{kind}:{window}"][model][s]["ap"]
                    for s in sensors
                ]),
            )

    evaluated: dict[str, dict] = {}
    variants = [(m, k, selected[m][k]) for m in MODELS for k in WINDOWS]
    variants += [(m, "base", 0) for m in MODELS]
    for model, kind, window in variants:
        name = f"{model}+{kind}"
        scorer = MODELS[model]
        by_sensor = []
        sensor_metrics = {}
        for sensor in sensors:
            frame = (
                bases[sensor]
                if kind == "base"
                else frame_with_feature(bases[sensor], kind, window)
            )
            scored, fit = scorer(frame, sensor, kind)
            alarmed = common.apply_past_thresholds(scored)
            later = alarmed.loc[EVALUATION_START:]
            if later.empty:
                raise ValueError(f"No later rows for {name}/{sensor}")
            sensor_metrics[sensor] = {
                "metrics": common.summarize(later)["1h"],
                "train_rows": fit["train_rows"],
            }
            by_sensor.append(later)
        pooled = common.summarize(pd.concat(by_sensor).sort_index())["1h"]
        evaluated[name] = {
            "window": window,
            "pooled_1h": pooled,
            "sensors": sensor_metrics,
        }
        print(
            f"LATER {name}({window}): AP={pooled['average_precision']:.5f} "
            f"F1={pooled['f1']:.5f}", flush=True,
        )

    report = {
        "protocol": {
            "data_step_minutes": 10,
            "model_fit": "June 13–21, 2025; normal targets and normal lag context",
            "development": "June 29–July 19, 2025; macro AP across three sensors",
            "later_evaluation": "July 20–September 8, 2025; not used in this window sweep",
            "threshold": "daily max F1 on preceding seven labelled days; q99 for normal-only history",
            "alarm": "six consecutive 10-minute threshold exceedances; causal",
            "caveat": "This later interval was visible in earlier research experiments, so it is not a never-seen independent test.",
        },
        "development_runs": runs,
        "selected_windows": selected,
        "later_evaluation": evaluated,
    }
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "window_sweep_and_later_evaluation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    print("RESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
