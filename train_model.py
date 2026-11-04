import os
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = "weather_dataset.parquet"
OUTPUT_DIR = Path("outputs")
TARGET = "rain_next_hour"
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]
TEST_START_DATE = "2023-01-01"
TUNING_ROWS = 50_000
SEARCH_ITERATIONS = 8
CV_FOLDS = 3
F1_TOLERANCE = 0.01
RANDOM_STATE = 42


def load_dataset(path):
    print(f"[1] Loading dataset: {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["time"])
    print(f"    {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_by_time(df):
    print(f"[2] Splitting: train = before {TEST_START_DATE}, test = from {TEST_START_DATE}")
    df = df.sort_values(["time", "city"]).reset_index(drop=True)
    train = df[df["time"] < TEST_START_DATE]
    test = df[df["time"] >= TEST_START_DATE]
    print(f"    train: {len(train)} rows | test: {len(test)} rows")
    print(f"    rain ratio  train: {train[TARGET].mean():.3f} | test: {test[TARGET].mean():.3f}")
    return train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]


def build_models(positive_weight):
    return {
        "LogisticRegression": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
            {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "max_depth": [5, 10, 15, 20],
                "min_samples_leaf": [1, 20, 100],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
            {
                "n_estimators": [50, 100],
                "max_depth": [8, 12, 16],
                "min_samples_leaf": [5, 20, 50],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                scale_pos_weight=positive_weight,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2],
            },
        ),
    }


def tune_hyperparameters(model, param_grid, X_train, y_train):
    step = max(1, len(X_train) // TUNING_ROWS)
    X_small = X_train.iloc[::step]
    y_small = y_train.iloc[::step]
    n_iter = min(SEARCH_ITERATIONS, len(ParameterGrid(param_grid)))
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=n_iter,
        scoring="f1",
        cv=TimeSeriesSplit(n_splits=CV_FOLDS),
        random_state=RANDOM_STATE,
    )
    search.fit(X_small, y_small)
    return search.best_params_, search.best_score_


def train_model(model, best_params, X_train, y_train):
    final_model = clone(model).set_params(**best_params)
    start = time.perf_counter()
    final_model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    return final_model, fit_seconds


def measure_prediction_time(model, X_test, repeats=10):
    sample = X_test.iloc[:1000]
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample)
    return (time.perf_counter() - start) / repeats * 1000


def save_model(model, name):
    path = OUTPUT_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path, os.path.getsize(path) / (1024 * 1024)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def run_all_models(X_train, y_train, X_test, y_test):
    positive_weight = (y_train == 0).sum() / (y_train == 1).sum()
    rows = []
    trained = {}
    for name, (model, param_grid) in build_models(positive_weight).items():
        print(f"\n--- {name} ---")
        print("    tuning hyperparameters ...")
        best_params, cv_f1 = tune_hyperparameters(model, param_grid, X_train, y_train)
        print(f"    best params: {best_params} (cv F1 = {cv_f1:.3f})")

        print("    training on full train set ...")
        final_model, fit_seconds = train_model(model, best_params, X_train, y_train)
        print(f"    fit time: {fit_seconds:.2f}s")

        metrics = evaluate_model(final_model, X_test, y_test)
        predict_ms = measure_prediction_time(final_model, X_test)
        model_path, size_mb = save_model(final_model, name)
        print(f"    test F1: {metrics['f1']:.3f} | saved: {model_path}")

        rows.append(
            {
                "algorithm": name,
                "fit_time_s": fit_seconds,
                "predict_1000_ms": predict_ms,
                "model_size_mb": size_mb,
                **metrics,
            }
        )
        trained[name] = final_model
    board = pd.DataFrame(rows).set_index("algorithm")
    return board.sort_values("f1", ascending=False), trained


def print_board(board, y_test):
    baseline = (y_test == 0).mean()
    print("\n" + "=" * 100)
    print("COMPARISON BOARD (test set, sorted by F1)")
    print("=" * 100)
    print(board.round(4).to_string())
    print("-" * 100)
    print(f"Baseline 'always predict no rain': accuracy = {baseline:.4f}, F1 = 0.0000")
    board.round(4).to_csv(OUTPUT_DIR / "comparison_board.csv")
    print(f"Board saved to {OUTPUT_DIR / 'comparison_board.csv'}")


def choose_best(board):
    top_f1 = board["f1"].max()
    candidates = board[board["f1"] >= top_f1 - F1_TOLERANCE]
    return candidates.sort_values(["model_size_mb", "predict_1000_ms"]).index[0]


def explain_choice(board, winner):
    best = board.loc[winner]
    print("\n" + "=" * 100)
    print(f"WHY CHOOSE {winner}?")
    print("=" * 100)
    print("Rule: the label is imbalanced, so accuracy is misleading -> rank by F1 first.")
    print(f"      If several models are within {F1_TOLERANCE} F1 of the top, pick the smallest/fastest one.")
    print(
        f"\n{winner}: F1 = {best['f1']:.4f}, ROC-AUC = {best['roc_auc']:.4f}, "
        f"fit = {best['fit_time_s']:.1f}s, predict 1000 rows = {best['predict_1000_ms']:.2f}ms, "
        f"size = {best['model_size_mb']:.2f}MB"
    )
    for name, row in board.iterrows():
        if name == winner:
            continue
        f1_gap = best["f1"] - row["f1"]
        if f1_gap > F1_TOLERANCE:
            reason = f"F1 is {f1_gap:.4f} lower"
        else:
            reason = (
                f"similar F1 (gap {f1_gap:+.4f}) but model is {row['model_size_mb']:.2f}MB "
                f"vs {best['model_size_mb']:.2f}MB and predicts in {row['predict_1000_ms']:.2f}ms "
                f"vs {best['predict_1000_ms']:.2f}ms"
            )
        print(f"  - not {name}: {reason}")


def show_winner_details(model, X_test, y_test, winner):
    matrix = confusion_matrix(y_test, model.predict(X_test))
    print(f"\nConfusion matrix of {winner} (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=["actual no rain", "actual rain"], columns=["pred no rain", "pred rain"]))
    joblib.dump(model, OUTPUT_DIR / "best_model.joblib")
    print(f"\nFinal model saved to {OUTPUT_DIR / 'best_model.joblib'}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_dataset(DATA_PATH)
    X_train, y_train, X_test, y_test = split_by_time(df)
    print("[3] Tuning, training and evaluating 4 algorithms")
    board, trained = run_all_models(X_train, y_train, X_test, y_test)
    print_board(board, y_test)
    winner = choose_best(board)
    explain_choice(board, winner)
    show_winner_details(trained[winner], X_test, y_test, winner)


if __name__ == "__main__":
    main()
