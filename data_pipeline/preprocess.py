"""
preprocess.py
─────────────
Step 2 of the pipeline.

Loads all raw CSV files, harmonises column names into the canonical
BEHAVIORAL_FEATURES schema, merges into a single training dataframe,
and performs cleaning + normalisation.

Output:
  data/processed/training_dataset.csv      ← unified feature matrix + target
  data/processed/work_style_dataset.csv    ← for work-style classifier
  data/processed/distraction_dataset.csv   ← for distraction scorer
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

warnings.filterwarnings("ignore")
console = Console()

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_DIR, PROCESSED_DIR,
    BEHAVIORAL_FEATURES, TARGET_FAILURE, TARGET_STYLE, RANDOM_STATE
)

rng = np.random.default_rng(RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════════════════
# Loaders — each maps source columns → canonical BEHAVIORAL_FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _fill_missing(df: pd.DataFrame, cols: list[str], mean_range=(0, 1)) -> pd.DataFrame:
    """Add canonical columns that weren't in the source, filling with plausible values."""
    for col in cols:
        if col not in df.columns:
            low, high = mean_range
            df[col] = rng.uniform(low, high, len(df))
    return df


def load_work_style(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        "daily_avg_hours":        "session_duration_minutes",   # × 60 below
        "procrastination_score":  "distraction_events",
        "self_reported_stress":   "stress_level",
        "completion_consistency": "previous_completion_rate",
        "peak_focus_duration_min": "focus_score",
        "task_switch_rate":       "tab_switches_proxy",
    }
    df = df.rename(columns=rename)
    df["session_duration_minutes"] = (df["session_duration_minutes"] * 60).clip(30, 720)
    df["break_count"] = (df.get("break_frequency_per_hour", 1) * (df["session_duration_minutes"] / 60)).round(0)
    df["social_media_minutes_before"] = rng.exponential(20, len(df)).clip(0, 120)
    df["task_complexity"]    = rng.integers(1, 6, len(df)).astype(float)
    df["time_of_day_hour"]   = rng.integers(6, 23, len(df)).astype(float)
    df["day_of_week"]        = rng.integers(0, 7, len(df)).astype(float)
    df["sleep_hours"]        = rng.normal(7, 1, len(df)).clip(4, 10)
    df["deadline_days_remaining"] = rng.exponential(3, len(df)).clip(0, 30)
    df["motivation_level"]   = rng.integers(1, 11, len(df)).astype(float)
    df["study_hours_weekly"] = rng.normal(25, 8, len(df)).clip(0, 70)
    df["work_style_score"]   = df["work_style_label"].map({"turtle": 0.1, "hare": 0.9, "hybrid": 0.5})
    return df


def load_study_habits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["session_duration_minutes"] = (df["study_hours_weekly"] / 7 * 60).clip(10, 480)
    df["break_count"] = rng.integers(0, 8, len(df)).astype(float)
    df["social_media_minutes_before"] = df["social_media_hours_daily"] * 20  # proxy
    df["task_complexity"] = rng.integers(1, 6, len(df)).astype(float)
    df["work_style_score"] = rng.uniform(0, 1, len(df))
    df["time_of_day_hour"] = rng.integers(6, 23, len(df)).astype(float)
    df["day_of_week"]      = rng.integers(0, 7, len(df)).astype(float)
    df["distraction_events"] = (df["social_media_hours_daily"] * 2).round(0)
    df["deadline_days_remaining"] = rng.exponential(4, len(df)).clip(0, 30)
    df["previous_completion_rate"] = rng.uniform(0.3, 1.0, len(df))
    df["focus_score"] = (
        df["study_hours_weekly"] / 70
        - df["stress_level"] / 40
        + df["motivation_level"] / 40
    ).clip(0, 1)
    df["work_style_label"] = "hybrid"   # placeholder; overridden during merge
    return df


def load_remote_worker(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "hours_logged" in df.columns:
        df["session_duration_minutes"] = (df["hours_logged"] * 60).clip(30, 720)
        df["distraction_events"] = df.get("tab_switches", rng.poisson(15, len(df)))
        df["previous_completion_rate"] = df.get("completion_rate", rng.uniform(0.4, 1.0, len(df)))
        df["focus_score"] = df.get("productivity_score", rng.uniform(0.3, 0.95, len(df)))
    else:
        # Real HF dataset — normalise whatever columns it has
        df["session_duration_minutes"] = rng.normal(360, 90, len(df)).clip(30, 720)
        df["distraction_events"]       = rng.poisson(15, len(df))
        df["previous_completion_rate"] = rng.uniform(0.4, 1.0, len(df))
        df["focus_score"]              = rng.uniform(0.3, 0.95, len(df))
        if "task_completed" not in df.columns:
            df["task_completed"] = rng.integers(0, 2, len(df))

    df["break_count"] = df.get("break_count", rng.integers(0, 6, len(df)).astype(float))
    df["social_media_minutes_before"] = rng.exponential(18, len(df)).clip(0, 120)
    df["task_complexity"]    = rng.integers(1, 6, len(df)).astype(float)
    df["work_style_score"]   = rng.uniform(0, 1, len(df))
    df["time_of_day_hour"]   = rng.integers(6, 23, len(df)).astype(float)
    df["day_of_week"]        = rng.integers(0, 7, len(df)).astype(float)
    df["stress_level"]       = rng.integers(1, 11, len(df)).astype(float)
    df["sleep_hours"]        = rng.normal(7, 1.2, len(df)).clip(4, 10)
    df["deadline_days_remaining"] = rng.exponential(3, len(df)).clip(0, 30)
    df["motivation_level"]   = rng.integers(1, 11, len(df)).astype(float)
    df["study_hours_weekly"] = rng.normal(30, 10, len(df)).clip(0, 70)
    df["work_style_label"]   = "hybrid"
    return df


def load_social_media(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["session_duration_minutes"] = rng.normal(360, 90, len(df)).clip(30, 720)
    df["break_count"]    = rng.integers(0, 6, len(df)).astype(float)
    df["social_media_minutes_before"] = df["pre_task_sm_minutes"]
    df["task_complexity"]    = rng.integers(1, 6, len(df)).astype(float)
    df["work_style_score"]   = rng.uniform(0, 1, len(df))
    df["time_of_day_hour"]   = rng.integers(6, 23, len(df)).astype(float)
    df["day_of_week"]        = rng.integers(0, 7, len(df)).astype(float)
    df["stress_level"]       = rng.integers(1, 11, len(df)).astype(float)
    df["sleep_hours"]        = rng.normal(7, 1, len(df)).clip(4, 10)
    df["distraction_events"] = df["phone_pickups"] / 10
    df["deadline_days_remaining"] = rng.exponential(3, len(df)).clip(0, 30)
    df["previous_completion_rate"] = rng.uniform(0.3, 1.0, len(df))
    df["focus_score"] = 1 - df["distraction_score"]
    df["motivation_level"]   = rng.integers(1, 11, len(df)).astype(float)
    df["study_hours_weekly"] = rng.normal(25, 8, len(df)).clip(0, 70)
    df["work_style_label"]   = "hybrid"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Merger
# ══════════════════════════════════════════════════════════════════════════════

def merge_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []

    loaders = {
        "work_style_dataset.csv":  load_work_style,
        "study_habits.csv":        load_study_habits,
        "remote_worker_productivity.csv": load_remote_worker,
        "remote_worker_productivity_synthetic.csv": load_remote_worker,
        "social_media_distraction.csv": load_social_media,
    }

    for fname, loader_fn in loaders.items():
        fpath = RAW_DIR / fname
        if fpath.exists():
            console.log(f"  Loading [bold]{fname}[/bold]…")
            try:
                df = loader_fn(fpath)
                frames.append(df)
                console.log(f"    → {len(df):,} rows")
            except Exception as e:
                console.log(f"  [red]  Failed to load {fname}: {e}[/red]")

    if not frames:
        raise RuntimeError("No raw data files found. Run generate_data.py first.")

    combined = pd.concat(frames, ignore_index=True)
    console.log(f"\n[bold]Combined raw rows:[/bold] {len(combined):,}")

    # ── Ensure all canonical features exist ───────────────────────────────
    for feat in BEHAVIORAL_FEATURES:
        if feat not in combined.columns:
            # Best-guess fill
            combined[feat] = rng.uniform(0, 1, len(combined))

    # ── Clip/type cleanup ─────────────────────────────────────────────────
    combined["session_duration_minutes"]   = combined["session_duration_minutes"].clip(10, 720)
    combined["break_count"]                = combined["break_count"].clip(0, 20)
    combined["social_media_minutes_before"]= combined["social_media_minutes_before"].clip(0, 180)
    combined["task_complexity"]            = combined["task_complexity"].clip(1, 5)
    combined["work_style_score"]           = combined["work_style_score"].clip(0, 1)
    combined["time_of_day_hour"]           = combined["time_of_day_hour"].clip(0, 23)
    combined["day_of_week"]                = combined["day_of_week"].clip(0, 6)
    combined["stress_level"]               = combined["stress_level"].clip(1, 10)
    combined["sleep_hours"]                = combined["sleep_hours"].clip(2, 12)
    combined["distraction_events"]         = combined["distraction_events"].clip(0, 50)
    combined["deadline_days_remaining"]    = combined["deadline_days_remaining"].clip(0, 90)
    combined["previous_completion_rate"]   = combined["previous_completion_rate"].clip(0, 1)
    combined["focus_score"]                = combined["focus_score"].clip(0, 1)
    combined["motivation_level"]           = combined["motivation_level"].clip(1, 10)
    combined["study_hours_weekly"]         = combined["study_hours_weekly"].clip(0, 84)

    # ── Target ────────────────────────────────────────────────────────────
    if TARGET_FAILURE not in combined.columns:
        combined[TARGET_FAILURE] = 1  # fallback
    combined[TARGET_FAILURE] = combined[TARGET_FAILURE].fillna(0).astype(int).clip(0, 1)

    if TARGET_STYLE not in combined.columns:
        combined[TARGET_STYLE] = "hybrid"
    combined[TARGET_STYLE] = combined[TARGET_STYLE].fillna("hybrid")

    combined = combined.dropna(subset=BEHAVIORAL_FEATURES + [TARGET_FAILURE])
    console.log(f"[green]After cleaning:[/green] {len(combined):,} rows")

    # ── Split outputs ─────────────────────────────────────────────────────
    training_cols = BEHAVIORAL_FEATURES + [TARGET_FAILURE, TARGET_STYLE]
    training_df   = combined[[c for c in training_cols if c in combined.columns]].copy()

    # Work-style subset (only rows with labelled style)
    ws_df = combined[combined[TARGET_STYLE].isin(["turtle", "hare", "hybrid"])].copy()

    # Distraction subset
    dm_df = combined[["distraction_events", "social_media_minutes_before",
                       "break_count", "session_duration_minutes",
                       "focus_score", TARGET_FAILURE]].copy()

    return training_df, ws_df, dm_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    console.print(Panel.fit("🔧 Step 2 — Preprocessing & Merging", style="bold magenta"))

    training_df, ws_df, dm_df = merge_all()

    # Save
    training_df.to_csv(PROCESSED_DIR / "training_dataset.csv", index=False)
    ws_df.to_csv(PROCESSED_DIR / "work_style_dataset.csv", index=False)
    dm_df.to_csv(PROCESSED_DIR / "distraction_dataset.csv", index=False)

    # Print summary table
    table = Table(title="Processed Datasets")
    table.add_column("File", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    table.add_row("training_dataset.csv", str(len(training_df)), str(len(training_df.columns)))
    table.add_row("work_style_dataset.csv", str(len(ws_df)), str(len(ws_df.columns)))
    table.add_row("distraction_dataset.csv", str(len(dm_df)), str(len(dm_df.columns)))
    console.print(table)

    # Label balance
    vc = training_df[TARGET_FAILURE].value_counts().to_dict()
    console.print(f"\nTask completion balance → Completed: {vc.get(1,0):,}  |  Failed: {vc.get(0,0):,}")

    console.print("\n[bold green]✅ Preprocessing complete → data/processed/[/bold green]")


if __name__ == "__main__":
    main()
