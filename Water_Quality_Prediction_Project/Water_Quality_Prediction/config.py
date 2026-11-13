from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "water_potability.csv"
TARGET_COLUMN = "Potability"

FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
MODELS_DIR = PROJECT_ROOT / "output" / "models"
METRICS_PATH = PROJECT_ROOT / "output" / "metrics.json"
MODEL_PATH = MODELS_DIR / "water_quality_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
