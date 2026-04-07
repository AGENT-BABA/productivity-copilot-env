"""
train_models.py
───────────────
Step 3 of the pipeline. Trains three models and saves all artifacts.

Models:
  A) Task Failure Predictor — XGBoost binary classifier
     Input : 15 behavioral features
     Output: failure_probability (0–1)
     Artifact: model_artifacts/failure_predictor.pkl

  B) Work Style Classifier — Random Forest 3-class
     Input : 7 work-style features
     Output: "turtle" | "hare" | "hybrid"
     Artifact: model_artifacts/work_style_classifier.pkl

  C) Distraction Scorer — Gradient Boosted Regressor
     Input : 5 distraction signals
     Output: distraction_score (0–1)
     Artifact: model_artifacts/distraction_scorer.pkl

  Also saves:
     model_artifacts/feature_scaler.pkl       ← StandardScaler fitted on training set
     model_artifacts/feature_columns.json     ← exact column lists for each model
     model_artifacts/metrics.json             ← evaluation metrics for dashboard display
"""

import sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    accuracy_score, confusion_matrix, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
console = Console()

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, ARTIFACTS_DIR,
    BEHAVIORAL_FEATURES, TARGET_FAILURE, TARGET_STYLE,
    XGBOOST_PARAMS, RF_PARAMS, RANDOM_STATE, TEST_SIZE
)

METRICS = {}   # filled during training, saved to JSON at end

# ══════════════════════════════════════════════════════════════════════════════
# A — Task Failure Predictor (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

def train_failure_predictor(df: pd.DataFrame) -> dict:
    console.rule("[bold cyan]A — Task Failure Predictor[/bold cyan]")

    feat_cols = [c for c in BEHAVIORAL_FEATURES if c in df.columns]
    X = df[feat_cols].values
    y = df[TARGET_FAILURE].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE to balance classes
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    console.log(f"After SMOTE → {np.bincount(y_res)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_res
    )

    # --- Train ---
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluate ---
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc  = roc_auc_score(y_test, y_proba)
    f1   = f1_score(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    console.print(classification_report(y_test, y_pred))
    console.log(f"[bold]AUC-ROC:[/bold] {auc:.4f}  |  [bold]F1:[/bold] {f1:.4f}  |  [bold]Accuracy:[/bold] {acc:.4f}")

    # 5-fold CV AUC on original (not resampled) to get honest estimate
    cv_model = XGBClassifier(**XGBOOST_PARAMS)
    cv_scaler = StandardScaler().fit(df[feat_cols].values)
    cv_scores = cross_val_score(
        cv_model,
        cv_scaler.transform(df[feat_cols].values),
        df[TARGET_FAILURE].values,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
    )
    console.log(f"5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance plot
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feat_cols, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title("XGBoost Feature Importances — Task Failure Predictor")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "feature_importance_failure.png", dpi=150)
    plt.close()

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted: Fail", "Predicted: Complete"],
                yticklabels=["Actual: Fail", "Actual: Complete"], ax=ax)
    ax.set_title("Confusion Matrix — Failure Predictor")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix_failure.png", dpi=150)
    plt.close()

    # --- Save ---
    joblib.dump(model,  ARTIFACTS_DIR / "failure_predictor.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "feature_scaler.pkl")
    console.log("[green]✓ Saved: failure_predictor.pkl, feature_scaler.pkl[/green]")

    return {
        "failure_predictor": {
            "auc_roc":  round(auc, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "cv_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_auc_std":  round(float(cv_scores.std()), 4),
            "n_train": int(len(X_train)),
            "n_test":  int(len(X_test)),
            "features": feat_cols,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# B — Work Style Classifier (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════

WORK_STYLE_FEATURES = [
    "session_duration_minutes",
    "break_count",
    "distraction_events",
    "stress_level",
    "motivation_level",
    "previous_completion_rate",
    "deadline_days_remaining",
]

def train_work_style_classifier(df: pd.DataFrame) -> dict:
    console.rule("[bold cyan]B — Work Style Classifier[/bold cyan]")

    feat_cols = [c for c in WORK_STYLE_FEATURES if c in df.columns]
    # Only rows with labelled style
    df_ws = df[df[TARGET_STYLE].isin(["turtle", "hare", "hybrid"])].copy()

    if len(df_ws) < 100:
        console.log("[yellow]⚠ Not enough labelled work-style rows. Skipping.[/yellow]")
        return {}

    le = LabelEncoder()
    y  = le.fit_transform(df_ws[TARGET_STYLE])
    X  = df_ws[feat_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")

    console.print(classification_report(y_test, y_pred, target_names=le.classes_))
    console.log(f"[bold]Accuracy:[/bold] {acc:.4f}  |  [bold]Macro-F1:[/bold] {f1:.4f}")

    # Save
    joblib.dump(model, ARTIFACTS_DIR / "work_style_classifier.pkl")
    joblib.dump(le,    ARTIFACTS_DIR / "work_style_label_encoder.pkl")
    console.log("[green]✓ Saved: work_style_classifier.pkl, work_style_label_encoder.pkl[/green]")

    return {
        "work_style_classifier": {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "classes":  list(le.classes_),
            "n_train": int(len(X_train)),
            "n_test":  int(len(X_test)),
            "features": feat_cols,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# C — Distraction Scorer (Gradient Boosting Regressor)
# ══════════════════════════════════════════════════════════════════════════════

DISTRACTION_FEATURES = [
    "distraction_events",
    "social_media_minutes_before",
    "break_count",
    "session_duration_minutes",
    "focus_score",
]

def train_distraction_scorer(df: pd.DataFrame) -> dict:
    console.rule("[bold cyan]C — Distraction Scorer[/bold cyan]")

    feat_cols = [c for c in DISTRACTION_FEATURES if c in df.columns]

    # Derive a distraction_score target from available signals
    df = df.copy()
    df["distraction_score"] = (
        0.35 * (df["distraction_events"] / df["distraction_events"].quantile(0.95)).clip(0, 1)
        + 0.30 * (df["social_media_minutes_before"] / 120).clip(0, 1)
        + 0.15 * (df["break_count"] / df["break_count"].quantile(0.95)).clip(0, 1)
        + 0.20 * (1 - df.get("focus_score", 0.5))
    ).clip(0, 1)

    X = df[feat_cols].values
    y = df["distraction_score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    console.log(f"[bold]RMSE:[/bold] {rmse:.4f}  |  [bold]R²:[/bold] {r2:.4f}")

    joblib.dump(model, ARTIFACTS_DIR / "distraction_scorer.pkl")
    console.log("[green]✓ Saved: distraction_scorer.pkl[/green]")

    return {
        "distraction_scorer": {
            "rmse": round(rmse, 4),
            "r2":   round(r2, 4),
            "n_train": int(len(X_train)),
            "n_test":  int(len(X_test)),
            "features": feat_cols,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# Save feature column manifest (backend uses this for inference)
# ══════════════════════════════════════════════════════════════════════════════

def save_feature_manifest(metrics: dict):
    manifest = {
        "failure_predictor": {
            "features": metrics.get("failure_predictor", {}).get("features", BEHAVIORAL_FEATURES),
            "threshold": 0.65,
        },
        "work_style_classifier": {
            "features": metrics.get("work_style_classifier", {}).get("features", WORK_STYLE_FEATURES),
            "classes": ["hare", "hybrid", "turtle"],
        },
        "distraction_scorer": {
            "features": metrics.get("distraction_scorer", {}).get("features", DISTRACTION_FEATURES),
        },
    }
    with open(ARTIFACTS_DIR / "feature_columns.json", "w") as f:
        json.dump(manifest, f, indent=2)
    console.log("[green]✓ Saved: feature_columns.json[/green]")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    console.print(Panel.fit("🤖 Step 3 — Model Training", style="bold magenta"))

    training_path = PROCESSED_DIR / "training_dataset.csv"
    if not training_path.exists():
        console.print("[red]❌ training_dataset.csv not found. Run preprocess.py first.[/red]")
        sys.exit(1)

    df = pd.read_csv(training_path)
    console.log(f"Loaded training data: {df.shape[0]:,} rows × {df.shape[1]} cols")

    metrics = {}

    # ── Train all three models ────────────────────────────────────────────────
    metrics.update(train_failure_predictor(df))
    metrics.update(train_work_style_classifier(df))
    metrics.update(train_distraction_scorer(df))

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    save_feature_manifest(metrics)
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    console.log("[green]✓ Saved: metrics.json[/green]")

    # ── Print summary ─────────────────────────────────────────────────────────
    console.print(Panel.fit("📊 Training Summary", style="bold green"))
    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Key Metric", style="yellow")
    table.add_column("Score", justify="right")

    fp = metrics.get("failure_predictor", {})
    ws = metrics.get("work_style_classifier", {})
    ds = metrics.get("distraction_scorer", {})

    if fp:
        table.add_row("Failure Predictor", "AUC-ROC", str(fp.get("auc_roc")))
        table.add_row("Failure Predictor", "F1 Score", str(fp.get("f1_score")))
        table.add_row("Failure Predictor", "CV AUC (5-fold)", f"{fp.get('cv_auc_mean')} ± {fp.get('cv_auc_std')}")
    if ws:
        table.add_row("Work Style Classifier", "Accuracy", str(ws.get("accuracy")))
        table.add_row("Work Style Classifier", "Macro-F1", str(ws.get("macro_f1")))
    if ds:
        table.add_row("Distraction Scorer", "RMSE", str(ds.get("rmse")))
        table.add_row("Distraction Scorer", "R²", str(ds.get("r2")))

    console.print(table)
    console.print("\n[bold green]✅ All model artifacts saved → model_artifacts/[/bold green]")


if __name__ == "__main__":
    main()
