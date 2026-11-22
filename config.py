from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "mental_wellbeing.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
TARGET_COLUMN = "Wellbeing_Score"
RANDOM_STATE = 42
TEST_SIZE = 0.20
