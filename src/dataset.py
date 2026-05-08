from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "population_a",
    "population_b",
    "food_total",
    "pollution",
    "avg_energy_a",
    "avg_energy_b",
    "birth_a",
    "birth_b",
    "death_a",
    "death_b",
]


def find_history_files(data_path):
    path = Path(data_path)

    if path.is_file():
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")

    history_files = sorted(path.rglob("history.csv"))

    if history_files:
        return history_files

    csv_files = sorted(path.rglob("*.csv"))

    if csv_files:
        return csv_files

    raise FileNotFoundError(f"No CSV files found in: {path}")


def validate_columns(df, file_path=None, feature_columns=None):
    columns = feature_columns or FEATURE_COLUMNS
    required = ["population_a"] + columns
    required = list(dict.fromkeys(required))

    missing = [column for column in required if column not in df.columns]

    if missing:
        location = f" in {file_path}" if file_path else ""
        raise ValueError(f"Missing columns{location}: {missing}")


def read_history_csv(file_path, feature_columns=None):
    path = Path(file_path)
    df = pd.read_csv(path)

    validate_columns(df, path, feature_columns)

    if "tick" in df.columns:
        df = df.sort_values("tick").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        df["tick"] = np.arange(len(df))

    return df


def load_runs(data_path, feature_columns=None):
    files = find_history_files(data_path)
    runs = []

    for file_path in files:
        df = read_history_csv(file_path, feature_columns)
        runs.append((file_path, df))

    return runs


def numeric_frame(df, feature_columns=None):
    columns = feature_columns or FEATURE_COLUMNS
    data = df[columns].apply(pd.to_numeric, errors="coerce")
    data = data.ffill().bfill().fillna(0.0)
    return data.to_numpy(dtype=float)


def fit_scaler(runs, feature_columns=None):
    columns = feature_columns or FEATURE_COLUMNS
    arrays = []

    for _, df in runs:
        arrays.append(numeric_frame(df, columns))

    values = np.vstack(arrays)

    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)

    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where((std == 0.0) | np.isnan(std), 1.0, std)

    return {
        "mean": mean,
        "std": std,
        "feature_columns": columns,
    }


def transform_frame(df, scaler):
    columns = scaler["feature_columns"]
    values = numeric_frame(df, columns)
    return (values - scaler["mean"]) / scaler["std"]


def make_samples_from_runs(
    runs,
    window_size,
    horizon,
    scaler,
    collapse_threshold=0.0,
):
    x_values = []
    y_values = []
    sample_ticks = []
    sample_sources = []

    for file_path, df in runs:
        features = transform_frame(df, scaler)
        population_a = pd.to_numeric(df["population_a"], errors="coerce")
        population_a = population_a.ffill().bfill().fillna(0.0).to_numpy(dtype=float)

        if "tick" in df.columns:
            ticks = pd.to_numeric(df["tick"], errors="coerce")
            ticks = ticks.ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        else:
            ticks = np.arange(len(df), dtype=float)

        max_start = len(df) - window_size - horizon + 1

        if max_start <= 0:
            continue

        for start in range(max_start):
            end = start + window_size
            future_end = end + horizon

            window = features[start:end]
            future_population = population_a[end:future_end]

            label = 1 if np.any(future_population <= collapse_threshold) else 0

            x_values.append(window)
            y_values.append(label)
            sample_ticks.append(ticks[end - 1])
            sample_sources.append(str(file_path))

    if not x_values:
        raise ValueError("No valid samples were created. Try reducing window_size or horizon.")

    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=int),
        np.asarray(sample_ticks, dtype=float),
        np.asarray(sample_sources, dtype=object),
    )


def split_samples(x, y, ticks, sources, test_ratio=0.2, seed=1):
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0.0, 1.0).")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)

    x = x[indices]
    y = y[indices]
    ticks = ticks[indices]
    sources = sources[indices]

    test_size = int(round(len(x) * test_ratio))

    if test_size == 0:
        return x, y, np.empty((0,) + x.shape[1:]), np.empty((0,), dtype=int), ticks, sources

    x_test = x[:test_size]
    y_test = y[:test_size]
    x_train = x[test_size:]
    y_train = y[test_size:]

    train_ticks = ticks[test_size:]
    train_sources = sources[test_size:]

    return x_train, y_train, x_test, y_test, train_ticks, train_sources


def load_dataset(
    data_path,
    window_size=50,
    horizon=100,
    test_ratio=0.2,
    seed=1,
    collapse_threshold=0.0,
    feature_columns=None,
):
    columns = feature_columns or FEATURE_COLUMNS
    runs = load_runs(data_path, columns)
    scaler = fit_scaler(runs, columns)

    x, y, ticks, sources = make_samples_from_runs(
        runs=runs,
        window_size=window_size,
        horizon=horizon,
        scaler=scaler,
        collapse_threshold=collapse_threshold,
    )

    x_train, y_train, x_test, y_test, train_ticks, train_sources = split_samples(
        x=x,
        y=y,
        ticks=ticks,
        sources=sources,
        test_ratio=test_ratio,
        seed=seed,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": columns,
        "window_size": window_size,
        "horizon": horizon,
        "collapse_threshold": collapse_threshold,
        "train_ticks": train_ticks,
        "train_sources": train_sources,
        "num_runs": len(runs),
        "num_samples": len(x),
    }