from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import catboost_variants as cbv
import ets
import evaluation_common as common
import sarima


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiment_rerun_2026-10-06"
LATER_START = pd.Timestamp("2025-07-20")


def main() -> None:
    values, labels = common.load_panel()
    sensors = [str(s) for s in values["io_id"].dropna().unique()]
    data: dict[str, list[pd.DataFrame]] = {
        "ETS": [], "SARIMA standardized": [], "SARIMA calibrated": [],
        "CatBoost original": [], "CatBoost sin/cos": [],
        "CatBoost + SR(t-1)": [], "CatBoost + SR fusion 25%": [],
    }
    fit_notes: dict[str, dict] = {}
    for sensor in sensors:
        ets_frame = ets.make_sensor_frame(values, labels, sensor)
        ets_scored = ets.daily_out_of_sample_scores(ets_frame, sensor, None)
        ets_alarm = ets.apply_past_thresholds(ets_scored).loc[LATER_START:]
        data["ETS"].append(ets_alarm.assign(alarm_1h=ets_alarm["causal_1h_prediction"]))
        print(f"{sensor[-2:]} ETS later done", flush=True)

        frame = common.make_sensor_frame(values, labels, sensor)
        sarima_scored, info = sarima.score_sensor(frame, sensor)
        sarima_alarm = common.apply_past_thresholds(sarima_scored)
        data["SARIMA standardized"].append(sarima_alarm.loc[LATER_START:])
        data["SARIMA calibrated"].append(
            sarima_alarm.assign(
                score=sarima_alarm["score"] / sarima_alarm["threshold"]
            ).loc[LATER_START:]
        )
        fit_notes[f"SARIMA {sensor}"] = {
            "unconverged_days": len(info["unconverged_days"]),
            "warning_days": len(info["warning_days"]),
        }
        print(f"{sensor[-2:]} SARIMA later done", flush=True)

        cb_frame = cbv.add_sr(cbv.add_calendar(
            cbv.baseline.make_sensor_frame(values, labels, sensor)
        ))
        variants = {
            "original": "CatBoost original",
            "cyclic": "CatBoost sin/cos",
            "cyclic_sr_lag1": "CatBoost + SR(t-1)",
        }
        for key, name in variants.items():
            scored, _ = cbv.score_model(cb_frame, sensor, key, None)
            alarm = common.apply_past_thresholds(scored)
            data[name].append(alarm.loc[LATER_START:])
            if key == "cyclic":
                fused = cbv.fuse_scores(scored, 0.25)
                data["CatBoost + SR fusion 25%"].append(fused.loc[LATER_START:])
        print(f"{sensor[-2:]} CatBoost variants later done", flush=True)

    metrics = {}
    for name, chunks in data.items():
        per_sensor = {
            sensor: common.metric_values(chunk, "alarm_1h")
            for sensor, chunk in zip(sensors, chunks)
        }
        metrics[name] = {
            "pooled": common.metric_values(pd.concat(chunks).sort_index(), "alarm_1h"),
            "macro": {
                field: float(np.mean([row[field] for row in per_sensor.values()]))
                for field in ("roc_auc", "average_precision", "precision", "recall", "f1")
            },
            "sensors": per_sensor,
        }
    report = {
        "protocol": {
            "evaluation_start": str(LATER_START.date()),
            "evaluation_end": "2025-09-08",
            "sampling": "10 minutes",
            "threshold": "previous seven labelled days only",
            "alarm": "six consecutive 10-minute threshold exceedances",
        },
        "metrics": metrics,
        "fit_notes": fit_notes,
    }
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "later_baselines.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    print("RESULT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
