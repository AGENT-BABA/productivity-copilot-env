"""
evaluate.py
───────────
Step 5 (optional). Loads saved model artifacts and runs a full evaluation report.
Saves plots to model_artifacts/ and prints a hackathon-ready summary.

Run after train_models.py.
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

from sklearn.metrics import (
    roc_curve, auc, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")
console = Console()

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, ARTIFACTS_DIR, BEHAVIORAL_FEATURES,
    TARGET_FAILURE, TARGET_STYLE, RANDOM_STATE, TEST_SIZE
)
from sklearn.model_selection import train_test_split


def load_artifacts():
    failure_model = joblib.load(ARTIFACTS_DIR / "failure_predictor.pkl")
    scaler        = joblib.load(ARTIFACTS_DIR / "feature_scaler.pkl")
    ws_model      = joblib.load(ARTIFACTS_DIR / "work_style_classifier.pkl")
    ws_encoder    = joblib.load(ARTIFACTS_DIR / "work_style_label_encoder.pkl")
    ds_model      = joblib.load(ARTIFACTS_DIR / "distraction_scorer.pkl")
    return failure_model, scaler, ws_model, ws_encoder, ds_model


def evaluate_failure_predictor(model, scaler, df):
    console.rule("[bold cyan]A — Failure Predictor Evaluation[/bold cyan]")
    feat_cols = [c for c in BEHAVIORAL_FEATURES if c in df.columns]
    X = scaler.transform(df[feat_cols].values)
    y = df[TARGET_FAILURE].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Task Failure Predictor — Evaluation Report", fontsize=14, fontweight="bold")

    # Plot ROC
    axes[0].plot(fpr, tpr, color="#00D4FF", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].set_facecolor("#0D1B2A")
    axes[0].figure.patch.set_alpha(0)

    # Plot PR
    axes[1].plot(recall, precision, color="#FF6B6B", lw=2, label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall Curve")
    axes[1].legend()

    # Plot score distribution
    axes[2].hist(y_proba[y_test == 0], bins=30, alpha=0.6, color="#FF6B6B", label="Failed Tasks")
    axes[2].hist(y_proba[y_test == 1], bins=30, alpha=0.6, color="#4ECDC4", label="Completed Tasks")
    axes[2].axvline(0.65, color="yellow", linestyle="--", label="Threshold (0.65)")
    axes[2].set_xlabel("Predicted Failure Probability")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Score Distribution")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "failure_predictor_eval.png", dpi=150, bbox_inches="tight")
    plt.close()
    console.log("[green]✓ Saved: failure_predictor_eval.png[/green]")

    console.print(classification_report(y_test, y_pred))
    console.log(f"ROC-AUC: {roc_auc:.4f} | Avg Precision: {ap:.4f}")


def evaluate_work_style(ws_model, ws_encoder, df):
    console.rule("[bold cyan]B — Work Style Classifier Evaluation[/bold cyan]")
    from config import RF_PARAMS
    WORK_STYLE_FEATURES = [
        "session_duration_minutes", "break_count", "distraction_events",
        "stress_level", "motivation_level", "previous_completion_rate",
        "deadline_days_remaining",
    ]
    df_ws = df[df[TARGET_STYLE].isin(["turtle", "hare", "hybrid"])].copy()
    if len(df_ws) < 50:
        console.log("[yellow]Insufficient labelled rows for work style evaluation.[/yellow]")
        return

    feat_cols = [c for c in WORK_STYLE_FEATURES if c in df_ws.columns]
    X = df_ws[feat_cols].values
    y = ws_encoder.transform(df_ws[TARGET_STYLE])
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    y_pred = ws_model.predict(X_test)

    console.print(classification_report(y_test, y_pred, target_names=ws_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ws_encoder.classes_,
                yticklabels=ws_encoder.classes_, ax=ax)
    ax.set_title("Work Style Classifier — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "work_style_confusion_matrix.png", dpi=150)
    plt.close()
    console.log("[green]✓ Saved: work_style_confusion_matrix.png[/green]")


def print_hackathon_summary():
    console.print(Panel.fit("🏆 Hackathon-Ready Model Summary", style="bold yellow"))
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if not metrics_path.exists():
        console.log("[red]metrics.json not found. Run train_models.py first.[/red]")
        return
    with open(metrics_path) as f:
        metrics = json.load(f)

    table = Table(title="Model Performance Metrics")
    table.add_column("Model",  style="cyan", min_width=30)
    table.add_column("Metric", style="white")
    table.add_column("Result", style="bold green", justify="right")

    fp = metrics.get("failure_predictor", {})
    ws = metrics.get("work_style_classifier", {})
    ds = metrics.get("distraction_scorer", {})

    if fp:
        table.add_row("Task Failure Predictor (XGBoost)", "AUC-ROC",    str(fp.get("auc_roc")))
        table.add_row("",                                  "F1 Score",   str(fp.get("f1_score")))
        table.add_row("",                                  "Accuracy",   str(fp.get("accuracy")))
        table.add_row("",                                  "5-fold CV AUC", f"{fp.get('cv_auc_mean')} ± {fp.get('cv_auc_std')}")
    if ws:
        table.add_row("Work Style Classifier (RF)",        "Accuracy",   str(ws.get("accuracy")))
        table.add_row("",                                  "Macro-F1",   str(ws.get("macro_f1")))
    if ds:
        table.add_row("Distraction Scorer (GBR)",          "RMSE",       str(ds.get("rmse")))
        table.add_row("",                                  "R²",         str(ds.get("r2")))

    console.print(table)
    console.print("\n[bold]Artifacts ready for backend integration:[/bold]")
    for f in sorted(ARTIFACTS_DIR.iterdir()):
        console.print(f"  [dim]→[/dim] {f.name}")


def main():
    console.print(Panel.fit("📊 Step 5 — Full Evaluation Report", style="bold magenta"))

    training_path = PROCESSED_DIR / "training_dataset.csv"
    if not training_path.exists():
        console.print("[red]Run generate_data.py and preprocess.py first.[/red]")
        sys.exit(1)

    df = pd.read_csv(training_path)
    failure_model, scaler, ws_model, ws_encoder, ds_model = load_artifacts()

    evaluate_failure_predictor(failure_model, scaler, df)
    evaluate_work_style(ws_model, ws_encoder, df)
    print_hackathon_summary()

    console.print("\n[bold green]✅ Evaluation complete. All plots saved to model_artifacts/[/bold green]")


if __name__ == "__main__":
    main()
