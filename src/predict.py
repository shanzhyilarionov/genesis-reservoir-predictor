import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import find_history_files, read_history_csv, transform_frame
from utils import format_percent, load_artifact, require_artifact_keys, risk_level


def make_prediction_windows(df, scaler, window_size, all_windows=False):
    features = transform_frame(df, scaler)

    if len(features) < window_size:
        raise ValueError(f"Input history has {len(features)} rows, but window_size is {window_size}.")

    if "tick" in df.columns:
        ticks = pd.to_numeric(df["tick"], errors="coerce")
        ticks = ticks.ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    else:
        ticks = np.arange(len(df), dtype=float)

    if not all_windows:
        return features[-window_size:][None, :, :], np.asarray([ticks[-1]])

    windows = []
    window_ticks = []

    for start in range(len(features) - window_size + 1):
        end = start + window_size
        windows.append(features[start:end])
        window_ticks.append(ticks[end - 1])

    return np.asarray(windows, dtype=float), np.asarray(window_ticks, dtype=float)


def predict_file(file_path, artifact, all_windows=False):
    df = read_history_csv(file_path, artifact["feature_columns"])

    x, ticks = make_prediction_windows(
        df=df,
        scaler=artifact["scaler"],
        window_size=artifact["window_size"],
        all_windows=all_windows,
    )

    risks = artifact["model"].predict_proba(x)

    results = pd.DataFrame(
        {
            "source": str(file_path),
            "tick": ticks,
            "collapse_risk": risks,
            "risk_level": [risk_level(risk) for risk in risks],
        }
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/reservoir_model.pkl")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    artifact = load_artifact(args.model)

    require_artifact_keys(
        artifact,
        [
            "model",
            "scaler",
            "feature_columns",
            "window_size",
            "horizon",
            "collapse_threshold",
        ],
    )

    files = find_history_files(args.input)
    frames = []

    for file_path in files:
        frames.append(predict_file(file_path, artifact, args.all))

    results = pd.concat(frames, ignore_index=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Saved predictions: {output_path}")
        return

    if args.all:
        for _, row in results.iterrows():
            print(
                f"{row['source']} | Tick: {int(row['tick'])} | "
                f"Collapse risk: {format_percent(row['collapse_risk'])} | "
                f"Risk level: {row['risk_level']}"
            )
        return

    for _, row in results.iterrows():
        print(f"Source: {row['source']}")
        print(f"Tick: {int(row['tick'])}")
        print(f"Collapse risk: {format_percent(row['collapse_risk'])}")
        print(f"Risk level: {row['risk_level']}")
        print("")


if __name__ == "__main__":
    main()