import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

from config import (
    DATA_PATH, TARGET_COLUMN, FIGURES_DIR, MODELS_DIR,
    METRICS_PATH, MODEL_PATH, METADATA_PATH
)
from src.data_utils import load_dataset, prepare_features_target
from src.model_utils import build_model
from src.visualization import save_eda_figures, save_evaluation_figures

def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Put your CSV file at data/water_potability.csv")
        sys.exit(1)

    df = load_dataset(DATA_PATH)
    X, y = prepare_features_target(df, TARGET_COLUMN)

    # Convert target labels to consistent strings/numbers as stored in CSV.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_eda_figures(df, X, FIGURES_DIR)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "recall_weighted": float(recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "f1_weighted": float(f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        ),
        "rows": int(len(df)),
        "features": list(X.columns)
    }

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save ranges and feature order for the Streamlit form.
    feature_info = {}
    for column in X.columns:
        series = X[column].dropna()
        feature_info[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median())
        }

    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_names": list(X.columns),
        "feature_info": feature_info,
        "class_labels": [str(v) for v in sorted(y.unique())]
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    save_evaluation_figures(model, X_test, y_test, FIGURES_DIR)

    print("\\nTraining completed!")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Graphs saved to: {FIGURES_DIR}")
    print(f"Metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    main()
