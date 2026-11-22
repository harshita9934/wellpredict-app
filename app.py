import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import os
from scipy.optimize import minimize

app = FastAPI(
    title="WellPredict Core API",
    description="Statistical ML Lifestyle Intelligence & SLSQP Optimization API",
    version="2.5.0"
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Model & Scaler Discovery -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
possible_model_paths = [
    os.path.join(BASE_DIR, "bayesian_model.pkl"),
    os.path.join(BASE_DIR, "Notebooks", "models", "bayesian_model.pkl"),
    os.path.join(BASE_DIR, "models", "bayesian_model.pkl"),
    os.path.join(BASE_DIR, "..", "Notebooks", "models", "bayesian_model.pkl"),
    os.path.join(BASE_DIR, "Mental_Wellbeing_Score_Project", "Notebooks", "models", "bayesian_model.pkl")
]
possible_scaler_paths = [
    os.path.join(BASE_DIR, "scaler.pkl"),
    os.path.join(BASE_DIR, "Notebooks", "models", "scaler.pkl"),
    os.path.join(BASE_DIR, "models", "scaler.pkl"),
    os.path.join(BASE_DIR, "..", "Notebooks", "models", "scaler.pkl"),
    os.path.join(BASE_DIR, "Mental_Wellbeing_Score_Project", "Notebooks", "models", "scaler.pkl")
]

model_path = next((p for p in possible_model_paths if os.path.exists(p)), None)
scaler_path = next((p for p in possible_scaler_paths if os.path.exists(p)), None)

if not model_path or not scaler_path:
    raise RuntimeError("Model ya Scaler file nahi mili! Check paths.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

FEATURE_NAMES = [
    'FRUITS_VEGGIES', 'DAILY_STRESS', 'PLACES_VISITED', 'CORE_CIRCLE',
    'DAILY_STEPS', 'SLEEP_HOURS', 'TIME_FOR_PASSION', 'WEEKLY_MEDITATION'
]

# ----------------- Schema -----------------
class LifestyleInput(BaseModel):
    FRUITS_VEGGIES: float
    DAILY_STRESS: float
    PLACES_VISITED: float
    CORE_CIRCLE: float
    DAILY_STEPS: float
    SLEEP_HOURS: float
    TIME_FOR_PASSION: float
    WEEKLY_MEDITATION: float

def to_vector(d: LifestyleInput):
    return np.array([[
        d.FRUITS_VEGGIES, d.DAILY_STRESS, d.PLACES_VISITED, d.CORE_CIRCLE,
        d.DAILY_STEPS, d.SLEEP_HOURS, d.TIME_FOR_PASSION, d.WEEKLY_MEDITATION
    ]])

# ----------------- Health Check -----------------
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "WellPredict API",
        "version": "2.5.0"
    }

# ----------------- 1. Prediction Confidence Engine -----------------
@app.post("/predict")
def predict_score(data: LifestyleInput):
    x = to_vector(data)
    x_scaled = scaler.transform(x)
    y_mean, y_std = model.predict(x_scaled, return_std=True)

    raw_score = float(y_mean[0])
    norm_score = np.clip((raw_score - 480.0) / (820.0 - 480.0) * 100.0, 0.0, 100.0)
    norm_std = (float(y_std[0]) / (820.0 - 480.0)) * 100.0

    ci_lower = float(np.clip(norm_score - 1.96 * norm_std, 0.0, 100.0))
    ci_upper = float(np.clip(norm_score + 1.96 * norm_std, 0.0, 100.0))

    if norm_score >= 70.0:
        status, color = "Optimal Equilibrium", "#10b981"
    elif norm_score >= 50.0:
        status, color = "Moderate Strain", "#f59e0b"
    else:
        status, color = "High Burnout Risk", "#ef4444"

    return {
        "score": round(float(norm_score), 1),
        "ci_lower": round(ci_lower, 1),
        "ci_upper": round(ci_upper, 1),
        "sigma": round(float(norm_std), 2),
        "status": status,
        "indicator_color": color
    }

# ----------------- 2. Explainable AI (XAI) Engine -----------------
@app.post("/explain")
def explain_features(data: LifestyleInput):
    x = to_vector(data)
    x_scaled = scaler.transform(x)[0]
    coefs = model.coef_

    contributions = x_scaled * coefs
    contrib_points = (contributions / (820.0 - 480.0)) * 100.0

    labels = {
        'FRUITS_VEGGIES': 'Nutrition Quality',
        'DAILY_STRESS': 'Daily Stress Level',
        'PLACES_VISITED': 'Exploration / Outing',
        'CORE_CIRCLE': 'Social Circle Buffer',
        'DAILY_STEPS': 'Physical Activity',
        'SLEEP_HOURS': 'Sleep Duration',
        'TIME_FOR_PASSION': 'Passion & Hobbies',
        'WEEKLY_MEDITATION': 'Mindfulness Practice'
    }

    feature_impacts = []
    for name, impact in zip(FEATURE_NAMES, contrib_points):
        feature_impacts.append({
            "feature": labels.get(name, name),
            "impact": round(float(impact), 1),
            "direction": "positive" if impact >= 0 else "negative"
        })

    feature_impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)

    return {
        "top_booster": next((f for f in feature_impacts if f["impact"] > 0), None),
        "top_drag": next((f for f in feature_impacts if f["impact"] < 0), None),
        "all_impacts": feature_impacts
    }

# ----------------- 3. 6-Month Trajectory Simulation -----------------
@app.post("/simulate")
def simulate_trajectory(data: LifestyleInput):
    base_res = predict_score(data)
    curr_score = base_res["score"]
    base_sigma = base_res["sigma"]

    timeline = ["Current", "Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]

    stress_penalty = max(0.0, (data.DAILY_STRESS - 2.0) * 1.6)
    sleep_deficit = max(0.0, (7.5 - data.SLEEP_HOURS) * 2.1)
    decay_rate = (stress_penalty + sleep_deficit) * 0.35

    status_quo, recovery, ci_upper, ci_lower = [], [], [], []

    for t in range(7):
        # Current Path (Degradation)
        sq = curr_score - (decay_rate * (t ** 1.12))
        sq_score = float(np.clip(sq, 15.0, 100.0))
        status_quo.append(round(sq_score, 1))

        # Recovery Path (Compound habit adoption toward target 82)
        rec = curr_score + (82.0 - curr_score) * (1.0 - np.exp(-0.55 * t))
        recovery.append(round(float(np.clip(rec, 0.0, 100.0)), 1))

        # Expanding Uncertainty
        s_t = base_sigma * np.sqrt(1.0 + 0.3 * t)
        ci_upper.append(round(float(np.clip(sq_score + 1.96 * s_t, 0.0, 100.0)), 1))
        ci_lower.append(round(float(np.clip(sq_score - 1.96 * s_t, 0.0, 100.0)), 1))

    return {
        "timeline": timeline,
        "status_quo": status_quo,
        "recovery": recovery,
        "ci_upper": ci_upper,
        "ci_lower": ci_lower
    }

# ----------------- 4. Minimum-Effort Optimization (SLSQP) -----------------
@app.post("/optimize")
def optimize_habits(data: LifestyleInput, target_score: float = 80.0):
    x0 = to_vector(data)[0]

    bounds = [
        (1.0, 5.0),   # Fruits
        (1.0, 5.0),   # Stress
        (0.0, 10.0),  # Places
        (1.0, 10.0),  # Circle
        (1.0, 10.0),  # Steps
        (5.0, 9.5),   # Sleep
        (0.5, 6.0),   # Passion
        (0.0, 10.0)   # Meditation
    ]

    weights = np.array([1.0, 3.5, 1.2, 1.0, 1.2, 1.5, 1.2, 1.0])
    raw_target = 480.0 + (target_score / 100.0) * (820.0 - 480.0)

    def objective(x):
        return np.sum(weights * ((x - x0) ** 2))

    def constraint(x):
        scaled = scaler.transform(x.reshape(1, -1))
        return model.predict(scaled)[0] - raw_target

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints={"type": "ineq", "fun": constraint})

    if res.success:
        opt = res.x
        return {
            "success": True,
            "target_score": target_score,
            "recommended": {
                "SLEEP_HOURS": round(float(opt[5]), 1),
                "DAILY_STRESS": round(float(opt[1]), 1),
                "WEEKLY_MEDITATION": round(float(opt[7]), 1),
                "TIME_FOR_PASSION": round(float(opt[6]), 1)
            },
            "shifts": {
                "sleep_delta": round(float(opt[5] - x0[5]), 1),
                "stress_delta": round(float(opt[1] - x0[1]), 1),
                "meditation_delta": round(float(opt[7] - x0[7]), 1)
            }
        }
    return {"success": False, "message": "Optimization did not converge within realistic bounds."}

# ----------------- 5. Clinical Health Diagnostics -----------------
@app.post("/diagnostics")
def get_diagnostics(data: LifestyleInput):
    risks = []
    if data.DAILY_STRESS >= 4.0 and data.SLEEP_HOURS < 6.5:
        risks.append({
            "title": "Chronic Cortisol Elevation & Executive Fog",
            "category": "Neuro-Cognitive Burnout",
            "severity": "High",
            "description": "Prolonged sleep deficit coupled with severe stress drives adrenal fatigue, reducing mental focus within 60 days."
        })
    elif data.DAILY_STRESS >= 4.0:
        risks.append({
            "title": "Sympathetic Nervous Overdrive",
            "category": "Mental Strain",
            "severity": "Moderate",
            "description": "Continuous high stress without decompression buffers depletes daily cognitive reserves."
        })

    if data.DAILY_STEPS <= 4.0 and data.FRUITS_VEGGIES <= 2.0:
        risks.append({
            "title": "Sedentary Metabolic Inertia",
            "category": "Physical Vitality",
            "severity": "Moderate",
            "description": "Deficient movement paired with low dietary antioxidants impairs daytime energy stability."
        })

    if data.CORE_CIRCLE <= 3.0 and data.PLACES_VISITED <= 2.0:
        risks.append({
            "title": "Support Network Depletion",
            "category": "Social Equilibrium",
            "severity": "Low",
            "description": "Limited social interaction amplifies vulnerability to acute academic/work stressors."
        })

    if not risks:
        risks.append({
            "title": "Homeostatic Equilibrium",
            "category": "Vitality",
            "severity": "Positive",
            "description": "Habit telemetry indicates balanced biological recovery and low burnout risk."
        })

    return {"diagnostics": risks}
