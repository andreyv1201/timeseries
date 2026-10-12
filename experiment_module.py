from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent
METHOD_SCRIPTS = {
    "catboost": "catboost_model.py",
    "catboost-variants": "catboost_variants.py",
    "xgboost": "xgboost_model.py",
    "ets": "ets.py",
    "matrix-profile": "matrix_profile.py",
    "sarima": "sarima.py",
    "sr-cnn": "srcnn.py",
    "representations-10min": "representation_experiments.py",
    "representations-1min": "minute_experiments.py",
    "window-sweep": "validation_sweep_2026_10_06.py",
    "later-baselines": "later_baselines_2026_10_06.py",
}
FILTERABLE = {
    "catboost", "catboost-variants", "xgboost", "ets", "matrix-profile",
    "sarima", "sr-cnn", "representations-10min", "representations-1min",
}
DATA_FILES = ("mv_cond_temperature.csv", "labled_mv_cond.csv")


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    methods: tuple[str, ...]
    sensor: str | None = None
    max_days: int | None = None
    matrix_min_points: int | None = None


@dataclass(frozen=True)
class ExperimentResult:
    method: str
    protocol: dict
    metrics: dict
    report: dict


def validate_config(config: ExperimentConfig) -> None:
    if not config.methods:
        raise ValueError("Select at least one method")
    unknown = set(config.methods) - METHOD_SCRIPTS.keys()
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(sorted(unknown))}")
    if config.max_days is not None and config.max_days < 1:
        raise ValueError("max_days must be positive")
    if config.matrix_min_points is not None and config.matrix_min_points < 1:
        raise ValueError("matrix_min_points must be positive")
    if config.matrix_min_points is not None and "matrix-profile" not in config.methods:
        raise ValueError("matrix_min_points requires matrix-profile")
    if (config.sensor is not None or config.max_days is not None) and any(
        method not in FILTERABLE for method in config.methods
    ):
        raise ValueError("sensor and max_days do not apply to the fixed-period comparisons")
    data_dir = config.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    for name in DATA_FILES:
        if not (data_dir / name).is_file():
            raise ValueError(f"Missing data file: {data_dir / name}")
    for method in config.methods:
        if not (ROOT / METHOD_SCRIPTS[method]).is_file():
            raise ValueError(f"Missing method file: {ROOT / METHOD_SCRIPTS[method]}")


def build_command(config: ExperimentConfig, method: str) -> list[str]:
    if method not in METHOD_SCRIPTS:
        raise ValueError(f"Unknown method: {method}")
    command = [sys.executable, str(ROOT / METHOD_SCRIPTS[method])]
    if config.sensor is not None:
        command.extend(("--sensor", config.sensor))
    if config.max_days is not None:
        command.extend(("--max-days", str(config.max_days)))
    if method == "matrix-profile" and config.matrix_min_points is not None:
        command.extend(("--min-anomaly-points", str(config.matrix_min_points)))
    return command


def extract_metrics(method: str, report: dict) -> dict:
    if method == "catboost":
        return {
            "raw": report["pooled_raw_threshold_metrics"],
            "2h": report["pooled_causal_2h_metrics"],
        }
    if method == "ets":
        return {
            "raw": report["pooled_raw_threshold_metrics"],
            "1h": report["pooled_causal_1h_metrics"],
            "12h": report["pooled_causal_12h_metrics"],
            "adaptive": report["pooled_adaptive_persistence_metrics"],
        }
    if method in {
        "xgboost", "sarima", "catboost-variants",
        "representations-10min", "representations-1min",
    }:
        return report["pooled"]
    if method in {"matrix-profile", "sr-cnn"}:
        metrics = {"raw": report["pooled_metrics"]}
        if "pooled_causal_1h_metrics" in report:
            metrics["1h"] = report["pooled_causal_1h_metrics"]
        if "late_holdout_pooled_metrics" in report:
            metrics["late"] = report["late_holdout_pooled_metrics"]
        return metrics
    if method == "window-sweep":
        return {
            name: entry["pooled_1h"]
            for name, entry in report["later_evaluation"].items()
        }
    if method == "later-baselines":
        return {
            name: entry["pooled"] for name, entry in report["metrics"].items()
        }
    raise ValueError(f"No metric adapter for {method}")


def run_method(
    config: ExperimentConfig,
    method: str,
    progress: Callable[[str], None] | None = print,
) -> ExperimentResult:
    validate_config(config)
    command = build_command(config, method)
    if progress is not None:
        progress(f"Running {method}")
    process = subprocess.Popen(
        command,
        cwd=config.data_dir.expanduser().resolve(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    report = None
    recent_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\r\n")
        if line.startswith("RESULT_JSON="):
            report = json.loads(line.removeprefix("RESULT_JSON="))
        else:
            recent_lines.append(line)
            recent_lines = recent_lines[-20:]
            if progress is not None:
                progress(line)
    status = process.wait()
    if status:
        raise RuntimeError(
            f"{method} failed with exit code {status}\n" + "\n".join(recent_lines)
        )
    if not isinstance(report, dict):
        raise RuntimeError(f"{method} completed without RESULT_JSON")
    return ExperimentResult(
        method=method,
        protocol=report.get("protocol", {}),
        metrics=extract_metrics(method, report),
        report=report,
    )


def run_experiments(
    config: ExperimentConfig,
    progress: Callable[[str], None] | None = print,
) -> dict[str, ExperimentResult]:
    validate_config(config)
    results = {}
    for method in dict.fromkeys(config.methods):
        results[method] = run_method(config, method, progress)
    return results


def metric_rows(metrics: dict, path: tuple[str, ...] = ()):
    if {"roc_auc", "average_precision", "f1"}.issubset(metrics):
        yield path, metrics
        return
    for name, value in metrics.items():
        if isinstance(value, dict):
            yield from metric_rows(value, (*path, name))


def print_summary(results: dict[str, ExperimentResult]) -> None:
    print("method\tvariant\tROC-AUC\tAP\tprecision\trecall\tF1")
    for method, result in results.items():
        for path, metrics in metric_rows(result.metrics):
            values = [metrics.get(key) for key in (
                "roc_auc", "average_precision", "precision", "recall", "f1"
            )]
            numbers = ["n/a" if value is None else f"{value:.4f}" for value in values]
            print("\t".join((method, "/".join(path), *numbers)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True, choices=(*METHOD_SCRIPTS, "all"))
    parser.add_argument("--sensor")
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--matrix-min-points", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    methods = tuple(METHOD_SCRIPTS) if "all" in args.methods else tuple(args.methods)
    config = ExperimentConfig(
        data_dir=args.data_dir,
        methods=methods,
        sensor=args.sensor,
        max_days=args.max_days,
        matrix_min_points=args.matrix_min_points,
    )
    try:
        validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        for method in dict.fromkeys(config.methods):
            print(subprocess.list2cmdline(build_command(config, method)))
        return
    results = run_experiments(config)
    print_summary(results)
    combined = {
        "methods": list(results),
        "metrics": {name: result.metrics for name, result in results.items()},
        "reports": {name: result.report for name, result in results.items()},
    }
    if args.output is not None:
        path = args.output.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
