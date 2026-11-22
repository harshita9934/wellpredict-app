import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent))
from config import DATA_PATH, MODEL_DIR, MODEL_PATH, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Put your CSV there first.")
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target '{TARGET_COLUMN}' not found. Available columns: {df.columns.tolist()}")
    df = df.dropna(subset=[TARGET_COLUMN])
    X, y = df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=8),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }
    results, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        results.append({"Model": name, "MAE": mean_absolute_error(y_test, pred), "MSE": mse, "RMSE": mse ** 0.5, "R2": r2_score(y_test, pred)})
        fitted[name] = pipe
    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(results_df.to_string(index=False))
    best_name = results_df.iloc[0]["Model"]
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_PATH)
    print(f"\nBest model: {best_name}\nSaved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
