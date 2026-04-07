"""
generate_data.py
────────────────
Step 1 of the pipeline.

Strategy (no Kaggle API required):
  1. Try to download free HuggingFace datasets (Remote Worker Productivity).
  2. Download Student Performance from OpenML API.
  3. For Kaggle-only datasets (Task Turtles, Study Habits, Social Media),
     generate statistically realistic synthetic data matching their schemas.

All outputs land in data/raw/ as CSV files.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

warnings.filterwarnings("ignore")
console = Console()

# ── Add parent to path so config.py is importable ─────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, SYNTHETIC_SEED, N_SYNTHETIC_SAMPLES

rng = np.random.default_rng(SYNTHETIC_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. HuggingFace – Remote Worker Productivity
# ══════════════════════════════════════════════════════════════════════════════
def download_remote_worker_hf() -> pd.DataFrame | None:
    """Attempt to load the Remote Worker Productivity dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        console.log("[cyan]Downloading Remote Worker Productivity from HuggingFace…[/cyan]")
        ds = load_dataset("nprak26/remote-worker-productivity", split="train")
        df = ds.to_pandas()
        out = RAW_DIR / "remote_worker_productivity.csv"
        df.to_csv(out, index=False)
        console.log(f"[green]✓ Saved {len(df):,} rows → {out.name}[/green]")
        return df
    except Exception as e:
        console.log(f"[yellow]⚠ HuggingFace download failed ({e}). Will use synthetic fallback.[/yellow]")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. UCI Student Performance — direct HTTP fetch (no openml library)
# ══════════════════════════════════════════════════════════════════════════════
def download_student_performance_openml() -> pd.DataFrame | None:
    """
    Fetch UCI Student Performance data directly from OpenML's REST API.
    Avoids the openml library which has a numpy>=2.x incompatibility.
    Falls back to synthetic data if the request fails.
    """
    try:
        console.log("[cyan]Fetching Student Performance data from OpenML API…[/cyan]")
        # Direct ARFF data endpoint for dataset 46589
        url = "https://api.openml.org/data/v1/download/22103147"
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code}")

        # Parse ARFF manually
        lines = resp.text.splitlines()
        attrs, data_lines = [], []
        in_data = False
        for line in lines:
            line = line.strip()
            if line.lower().startswith("@attribute"):
                parts = line.split()
                attrs.append(parts[1].strip("'\""))
            elif line.lower() == "@data":
                in_data = True
            elif in_data and line and not line.startswith("%"):
                data_lines.append(line.split(","))

        df = pd.DataFrame(data_lines, columns=attrs)
        out = RAW_DIR / "student_performance_uci.csv"
        df.to_csv(out, index=False)
        console.log(f"[green]✓ Saved {len(df):,} rows → {out.name}[/green]")
        return df
    except Exception as e:
        console.log(f"[yellow]⚠ OpenML fetch failed ({e}). Generating synthetic student data.[/yellow]")
        return generate_student_performance_synthetic()


def generate_student_performance_synthetic(n: int = 649) -> pd.DataFrame:
    """Synthetic version of UCI Student Performance dataset (Math subject)."""
    console.log("[cyan]Generating synthetic UCI Student Performance data…[/cyan]")
    studytime    = rng.integers(1, 5, n)
    failures     = rng.integers(0, 4, n)
    absences     = rng.integers(0, 75, n)
    age          = rng.integers(15, 23, n)
    famrel       = rng.integers(1, 6, n)
    freetime     = rng.integers(1, 6, n)
    goout        = rng.integers(1, 6, n)
    Dalc         = rng.integers(1, 6, n)
    health       = rng.integers(1, 6, n)
    G1           = rng.integers(0, 21, n).astype(float)
    G2           = (G1 + rng.integers(-3, 4, n)).clip(0, 20).astype(float)
    G3           = (G2 + rng.integers(-2, 3, n)).clip(0, 20).astype(float)

    df = pd.DataFrame({
        "age": age, "studytime": studytime, "failures": failures,
        "absences": absences, "famrel": famrel, "freetime": freetime,
        "goout": goout, "Dalc": Dalc, "health": health,
        "G1": G1, "G2": G2, "G3": G3,
        "task_completed": (G3 >= 10).astype(int),
    })
    out = RAW_DIR / "student_performance_uci.csv"
    df.to_csv(out, index=False)
    console.log(f"[green]✓ Saved {len(df):,} rows → {out.name}[/green]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Synthetic – Task Turtles vs Sprint Hares (Work Style)
# ══════════════════════════════════════════════════════════════════════════════
def generate_work_style_dataset(n: int = 2500) -> pd.DataFrame:
    """
    Mimics the schema of the Kaggle 'Task Turtles vs Sprint Hares' dataset.
    Three work styles: turtle (slow/steady), hare (fast/burst), hybrid.
    """
    console.log("[cyan]Generating synthetic Work Style dataset…[/cyan]")

    styles = rng.choice(["turtle", "hare", "hybrid"], size=n, p=[0.38, 0.35, 0.27])
    records = []

    for style in styles:
        if style == "turtle":
            rec = {
                "daily_avg_hours": rng.normal(6.5, 1.2),
                "peak_focus_duration_min": rng.normal(45, 12),
                "break_frequency_per_hour": rng.normal(0.9, 0.3),
                "deadline_lead_days": rng.normal(5.5, 2.0),
                "task_switch_rate": rng.normal(1.2, 0.5),
                "completion_consistency": rng.normal(0.82, 0.10),
                "overtime_events_per_week": rng.poisson(0.5),
                "procrastination_score": rng.normal(3.2, 1.1),  # 1-10
                "self_reported_stress": rng.normal(4.1, 1.5),
            }
        elif style == "hare":
            rec = {
                "daily_avg_hours": rng.normal(9.2, 1.8),
                "peak_focus_duration_min": rng.normal(90, 25),
                "break_frequency_per_hour": rng.normal(0.3, 0.2),
                "deadline_lead_days": rng.normal(0.8, 1.0),
                "task_switch_rate": rng.normal(3.5, 1.0),
                "completion_consistency": rng.normal(0.61, 0.15),
                "overtime_events_per_week": rng.poisson(2.8),
                "procrastination_score": rng.normal(7.5, 1.2),
                "self_reported_stress": rng.normal(7.2, 1.6),
            }
        else:  # hybrid
            rec = {
                "daily_avg_hours": rng.normal(7.5, 1.5),
                "peak_focus_duration_min": rng.normal(60, 18),
                "break_frequency_per_hour": rng.normal(0.6, 0.25),
                "deadline_lead_days": rng.normal(2.8, 1.5),
                "task_switch_rate": rng.normal(2.2, 0.8),
                "completion_consistency": rng.normal(0.72, 0.12),
                "overtime_events_per_week": rng.poisson(1.5),
                "procrastination_score": rng.normal(5.2, 1.3),
                "self_reported_stress": rng.normal(5.5, 1.5),
            }

        rec["work_style_label"] = style
        # Derived task_completed
        base_prob = {"turtle": 0.80, "hare": 0.55, "hybrid": 0.68}[style]
        stress_penalty = max(0, (rec["self_reported_stress"] - 5) * 0.03)
        rec["task_completed"] = int(rng.random() < (base_prob - stress_penalty))
        records.append(rec)

    df = pd.DataFrame(records)
    # Only clip numeric columns; string columns (work_style_label) must be excluded
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].clip(lower=0)

    out = RAW_DIR / "work_style_dataset.csv"
    df.to_csv(out, index=False)
    console.log(f"[green]✓ Saved {len(df):,} rows → {out.name}[/green]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Synthetic – Student Study Habits
# ══════════════════════════════════════════════════════════════════════════════
def generate_study_habits_dataset(n: int = 2000) -> pd.DataFrame:
    """Mimics Kaggle 'Student Study Habits' + 'Study Habits and Activities'."""
    console.log("[cyan]Generating synthetic Study Habits dataset…[/cyan]")

    study_hours     = rng.normal(4.5, 2.0, n).clip(0, 12)
    sleep_hours     = rng.normal(7.0, 1.2, n).clip(4, 10)
    social_media_h  = rng.exponential(2.5, n).clip(0, 10)
    stress          = rng.integers(1, 11, n).astype(float)
    motivation      = rng.integers(1, 11, n).astype(float)
    exercise_days   = rng.integers(0, 7, n).astype(float)
    attendance_pct  = rng.normal(80, 12, n).clip(40, 100)
    assignment_late = rng.poisson(1.2, n).clip(0, 8)

    # GPA as a function of the above
    gpa = (
        0.28 * study_hours
        + 0.10 * sleep_hours
        - 0.12 * social_media_h
        - 0.05 * stress
        + 0.08 * motivation
        + 0.04 * exercise_days
        + 0.01 * attendance_pct / 10
        - 0.06 * assignment_late
        + rng.normal(0, 0.25, n)
    ).clip(0, 4.0)

    task_completed = (gpa >= 2.5).astype(int)

    df = pd.DataFrame({
        "study_hours_weekly": study_hours * 7,
        "sleep_hours":        sleep_hours,
        "social_media_hours_daily": social_media_h,
        "stress_level":       stress,
        "motivation_level":   motivation,
        "exercise_days_weekly": exercise_days,
        "attendance_pct":     attendance_pct,
        "assignment_late_count": assignment_late,
        "gpa":                gpa.round(2),
        "task_completed":     task_completed,
    })

    out = RAW_DIR / "study_habits.csv"
    df.to_csv(out, index=False)
    console.log(f"[green]✓ Saved {len(df):,} rows → {out.name}[/green]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. Synthetic – Remote Worker Productivity (fallback)
# ══════════════════════════════════════════════════════════════════════════════
def generate_remote_worker_dataset(n: int = 2000) -> pd.DataFrame:
    """Synthetic fallback matching 'remote-worker-productivity' schema."""
    console.log("[cyan]Generating synthetic Remote Worker Productivity dataset…[/cyan]")

    hours_logged    = rng.normal(7.5, 1.8, n).clip(2, 14)
    meeting_hours   = rng.normal(2.0, 1.0, n).clip(0, 6)
    deep_work_hours = (hours_logged - meeting_hours - rng.normal(1.5, 0.5, n)).clip(0)
    tab_switches    = rng.poisson(18, n)
    idle_minutes    = rng.exponential(25, n).clip(0, 180)
    tasks_planned   = rng.integers(3, 12, n)
    tasks_done      = np.array([
        rng.integers(0, int(tp) + 1) for tp in tasks_planned.tolist()
    ])

    completion_rate = (tasks_done / tasks_planned).clip(0, 1)
    productivity_score = (
        0.4 * completion_rate
        + 0.3 * (deep_work_hours / (deep_work_hours + idle_minutes / 60 + 0.01)).clip(0, 1)
        - 0.2 * (tab_switches / 50).clip(0, 1)
        + rng.normal(0, 0.05, n)
    ).clip(0, 1)

    tasks_planned = tasks_planned.astype(float)
    tasks_done    = tasks_done.astype(float)
    task_completed = (completion_rate >= 0.6).astype(int)

    df = pd.DataFrame({
        "hours_logged":        hours_logged.round(1),
        "meeting_hours":       meeting_hours.round(1),
        "deep_work_hours":     deep_work_hours.round(1),
        "tab_switches":        tab_switches,
        "idle_minutes":        idle_minutes.round(1),
        "tasks_planned":       tasks_planned.astype(int),
        "tasks_completed":     tasks_done.astype(int),
        "completion_rate":     completion_rate.round(3),
        "productivity_score":  productivity_score.round(3),
        "task_completed":      task_completed,
    })

    out = RAW_DIR / "remote_worker_productivity_synthetic.csv"
    df.to_csv(out, index=False)
    console.log(f"[green]✓ Saved {len(df):,} rows → {out.name}[/green]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. Synthetic – Social Media & Distraction Patterns
# ══════════════════════════════════════════════════════════════════════════════
def generate_social_media_distraction_dataset(n: int = 1500) -> pd.DataFrame:
    """Mimics 'Social Media & Academic Performance' + 'Time Wasters on Social Media'."""
    console.log("[cyan]Generating synthetic Social Media Distraction dataset…[/cyan]")

    platforms = rng.choice(
        ["Instagram", "TikTok", "YouTube", "Twitter", "WhatsApp", "Reddit"],
        size=n
    )
    daily_sm_hours  = rng.exponential(3.0, n).clip(0, 12)
    notification_count = rng.poisson(45, n)
    phone_pickups   = rng.poisson(60, n)
    pre_task_sm_min = rng.exponential(20, n).clip(0, 120)
    binge_events    = rng.poisson(1.8, n)

    distraction_score = (
        0.30 * (daily_sm_hours / 12)
        + 0.25 * (notification_count / 100).clip(0, 1)
        + 0.25 * (phone_pickups / 150).clip(0, 1)
        + 0.20 * (pre_task_sm_min / 120)
    ).clip(0, 1)

    task_completed = (distraction_score < 0.45).astype(int)
    # add some noise
    flip_mask = rng.random(n) < 0.08
    task_completed[flip_mask] = 1 - task_completed[flip_mask]

    df = pd.DataFrame({
        "primary_platform":    platforms,
        "daily_sm_hours":      daily_sm_hours.round(1),
        "notification_count":  notification_count,
        "phone_pickups":       phone_pickups,
        "pre_task_sm_minutes": pre_task_sm_min.round(1),
        "binge_events":        binge_events,
        "distraction_score":   distraction_score.round(3),
        "task_completed":      task_completed,
    })

    out = RAW_DIR / "social_media_distraction.csv"
    df.to_csv(out, index=False)
    console.log(f"[green]✓ Saved {len(df):,} rows → {out.name}[/green]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    console.print(Panel.fit("📦 Step 1 — Data Generation & Download", style="bold magenta"))

    # 1. Try real HF dataset; fall back to synthetic
    rw_df = download_remote_worker_hf()
    if rw_df is None:
        rw_df = generate_remote_worker_dataset(n=2000)

    # 2. OpenML student performance
    uci_df = download_student_performance_openml()

    # 3. Synthetic datasets (no Kaggle API needed)
    ws_df  = generate_work_style_dataset(n=2500)
    sh_df  = generate_study_habits_dataset(n=2000)
    sm_df  = generate_social_media_distraction_dataset(n=1500)

    # Summary
    console.print("\n[bold green]✅ All datasets ready in data/raw/[/bold green]")
    sizes = {
        "remote_worker": len(rw_df),
        "work_style":    len(ws_df),
        "study_habits":  len(sh_df),
        "social_media":  len(sm_df),
    }
    if uci_df is not None:
        sizes["uci_student_performance"] = len(uci_df)
    console.print(json.dumps(sizes, indent=2))


if __name__ == "__main__":
    main()
