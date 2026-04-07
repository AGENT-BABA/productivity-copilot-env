"""
config.py — Central configuration for all pipeline scripts.
"""
import os
from pathlib import Path

# ── Root Paths ────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
ARTIFACTS_DIR  = ROOT_DIR / "model_artifacts"
VECTORSTORE_DIR= ROOT_DIR / "vectorstore"
NOTEBOOKS_DIR  = ROOT_DIR / "notebooks"

# Create all directories if they don't exist
for path in [RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, VECTORSTORE_DIR, NOTEBOOKS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ── Model Config ──────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
TEST_SIZE        = 0.20
VAL_SIZE         = 0.10
FAILURE_THRESHOLD= 0.65   # risk score >= this triggers an intervention

# XGBoost Hyperparameters (tuned for small-medium sized tabular behavioral data)
XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.5,   # slight adjustment for class imbalance
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}

# Work-Style Classifier (Random Forest)
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# ── LLM / RAG Config ──────────────────────────────────────────────────────────
GROQ_MODEL       = "llama-3.3-70b-versatile"   # free tier
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"          # local, no API needed
CHUNK_SIZE       = 400
CHUNK_OVERLAP    = 60
TOP_K_RETRIEVAL  = 5

# ── Synthetic Data Config ─────────────────────────────────────────────────────
N_SYNTHETIC_SAMPLES = 8000   # total rows across all synthetic sources
SYNTHETIC_SEED      = 42

# ── Feature Columns ───────────────────────────────────────────────────────────
BEHAVIORAL_FEATURES = [
    "session_duration_minutes",
    "break_count",
    "social_media_minutes_before",
    "task_complexity",          # 1–5
    "work_style_score",         # 0=turtle … 1=hare
    "time_of_day_hour",
    "day_of_week",
    "stress_level",             # 1–10
    "sleep_hours",
    "distraction_events",
    "deadline_days_remaining",
    "previous_completion_rate", # 0–1
    "focus_score",              # composite
    "motivation_level",         # 1–10
    "study_hours_weekly",
]
TARGET_FAILURE  = "task_completed"            # 1 = completed, 0 = failed/abandoned
TARGET_STYLE    = "work_style_label"          # "turtle" | "hare" | "hybrid"
