import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import download_data, validate_frame
from .evaluation import external_evaluate, run_experiment
from .explain import explain_records


def main():
    parser = argparse.ArgumentParser(description="Heart disease research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the original Kaggle CSV")
    download.add_argument("--output", default="data/raw/heart.csv")
    train = commands.add_parser("train", help="Run nested validation and fit the demonstration model")
    train.add_argument("--data", default="data/raw/heart.csv")
    train.add_argument("--output", default="artifacts/full")
    train.add_argument("--folds", type=int, choices=range(2, 11), default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--jobs", type=int, default=1)
    train.add_argument("--bootstrap", type=int, default=1000)
    train.add_argument("--smoke", action="store_true")
    external = commands.add_parser("external", help="Evaluate a frozen model on compatible independent data")
    external.add_argument("--model", default="artifacts/full/model.joblib")
    external.add_argument("--data", required=True)
    external.add_argument("--output", default="reports/external.json")
    predict = commands.add_parser("predict", help="Batch prediction, optional SHAP explanations")
    predict.add_argument("--model", default="artifacts/full/model.joblib")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", default="reports/predictions.csv")
    predict.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        print(download_data(args.output))
    elif args.command == "train":
        if args.bootstrap < 10:
            parser.error("--bootstrap must be at least 10")
        run_experiment(args.data, args.output, args.folds, args.seed, args.jobs, args.bootstrap, args.smoke)
    elif args.command == "external":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(external_evaluate(args.model, args.data, args.output), indent=2))
    else:
        bundle = joblib.load(args.model)
        frame = validate_frame(pd.read_csv(args.input))
        result = frame.copy()
        result["model_probability"] = bundle["model"].predict_proba(frame)[:, 1]
        result["model_class"] = (result.model_probability >= bundle["threshold"]).astype(int)
        result["decision_threshold"] = bundle["threshold"]
        if args.explain:
            explanation = explain_records(bundle, frame)
            for i, name in enumerate(explanation["features"]):
                result[f"shap_{name}"] = explanation["values"][:, i]
            result["shap_base"] = explanation["base_value"]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(args.output)


if __name__ == "__main__":
    main()
