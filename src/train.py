import argparse

import numpy as np

from dataset import FEATURE_COLUMNS, load_dataset
from evaluate import classification_metrics, print_metrics
from reservoir import EchoStateNetwork
from utils import save_artifact, utc_timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-output", default="models/reservoir_model.pkl")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--collapse-threshold", type=float, default=0.0)
    parser.add_argument("--reservoir-size", type=int, default=300)
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--input-scale", type=float, default=0.5)
    parser.add_argument("--leak-rate", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    dataset = load_dataset(
        data_path=args.data,
        window_size=args.window_size,
        horizon=args.horizon,
        test_ratio=args.test_ratio,
        seed=args.seed,
        collapse_threshold=args.collapse_threshold,
        feature_columns=FEATURE_COLUMNS,
    )

    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]

    if len(np.unique(y_train)) < 2:
        print("Warning: training labels contain only one class.")

    model = EchoStateNetwork(
        input_dim=x_train.shape[2],
        reservoir_size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        input_scale=args.input_scale,
        leak_rate=args.leak_rate,
        density=args.density,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
    )

    model.fit(x_train, y_train)

    train_risks = model.predict_proba(x_train)
    train_metrics = classification_metrics(y_train, train_risks)

    print("Training metrics")
    print_metrics(train_metrics)

    test_metrics = None

    if len(x_test) > 0:
        test_risks = model.predict_proba(x_test)
        test_metrics = classification_metrics(y_test, test_risks)

        print("")
        print("Test metrics")
        print_metrics(test_metrics)

    artifact = {
        "model": model,
        "scaler": dataset["scaler"],
        "feature_columns": dataset["feature_columns"],
        "window_size": dataset["window_size"],
        "horizon": dataset["horizon"],
        "collapse_threshold": dataset["collapse_threshold"],
        "created_at": utc_timestamp(),
        "num_runs": dataset["num_runs"],
        "num_samples": dataset["num_samples"],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "parameters": {
            "reservoir_size": args.reservoir_size,
            "spectral_radius": args.spectral_radius,
            "input_scale": args.input_scale,
            "leak_rate": args.leak_rate,
            "density": args.density,
            "ridge_alpha": args.ridge_alpha,
            "seed": args.seed,
        },
    }

    save_artifact(artifact, args.model_output)

    print("")
    print(f"Saved model: {args.model_output}")


if __name__ == "__main__":
    main()