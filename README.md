# Productivity Copilot — OpenEnv Environment

An AI agent simulation environment where an LLM agent acts as a **productivity coach** managing a virtual human worker. The agent must observe behaviour signals and take corrective actions to prevent task failure — powered by real trained machine learning models.

---

## Environment Description & Motivation

Modern knowledge workers face productivity challenges driven by distraction, stress, and poor time management. Instead of a toy environment, this simulation models **real-world task management** where an AI agent must intervene intelligently.

The agent is given a virtual human with observable state signals (stress level, distraction score, focus score, deadline pressure) and must apply targeted interventions. The simulation is grounded in real ML models trained on productivity behaviour data.

---

## Observation Space

Each observation is a `ProductivityObservation` Pydantic model:

| Field | Type | Description |
|---|---|---|
| `current_task` | str | The task the virtual human is working on |
| `deadline_days_remaining` | float | Days left until the task deadline |
| `stress_level` | float (0–10) | Current stress level |
| `motivation_level` | float (0–10) | Current motivation level |
| `distraction_events` | int | Count of distraction interruptions |
| `focus_score` | float (0–1) | Computed by the distraction scorer ML model |
| `failure_probability` | float (0–1) | Computed by the failure predictor ML model |
| `session_duration_minutes` | int | Minutes since last reset/break |
| `break_count` | int | Number of breaks taken |
| `social_media_minutes` | int | Minutes of social media use |
| `time_of_day_hour` | float | Current simulated hour of the day |

---

## Action Space

Each action is a `ProductivityAction` Pydantic model:

| `action_type` | Effect on Environment |
|---|---|
| `WAIT` | Time passes; stressed workers get worse |
| `SEND_NUDGE` | +2 motivation, -0.5 stress, -1 distraction |
| `FORCE_BREAK` | +1 break, session resets, -2 stress, +5 social media |
| `BLOCK_SOCIAL_MEDIA` | Social media set to 0, -3 distractions, +1 stress |

---

## Task Descriptions

### Task 1 — Triage (Easy)
A high-stress worker with a looming 1-day deadline. They have accumulated 10 distraction events and low motivation. The agent must identify the right intervention to lower failure probability in a single episode.
- **Objective:** Finish with `failure_probability < 0.5`

### Task 2 — Schedule Optimisation (Medium)
A "turtle" work-style employee (slow and steady) with only 0.5 days left on a complex task. The challenge is preventing failure without pushing stress above 8.
- **Objective:** Lower failure probability while keeping `stress_level < 8`

### Task 3 — Distraction Mitigation (Hard)
A "hare" worker who binge-works but is caught in extreme distraction (20 events). The agent must maintain `focus_score < 0.5` over the full 10-step episode despite the environment constantly generating more distractions.
- **Objective:** Keep average `focus_score < 0.5` across all steps

---

## Setup & Usage

### Local setup
```bash
# Create a virtual environment and install dependencies
pip install uv
uv sync

# Run openenv validate to confirm environment compliance
openenv validate
```

### Run the baseline agent
```bash
# Set your API credentials
export HF_TOKEN=your_api_key_here
export PRODUCTIVITY_TASK=triage  # or: schedule_optimization, distraction_mitigation

# Run the inference script
python inference.py
```

### Docker
```bash
docker build -t productivity-copilot-env .
docker run -p 7860:7860 -e HF_TOKEN=your_key productivity-copilot-env
```

---

## Baseline Scores

The baseline agent uses `Qwen/Qwen2.5-72B-Instruct` via the HuggingFace Router API.

| Task | Score | Notes |
|---|---|---|
| Task 1 — Triage | ~0.60 | Agent correctly prioritises SEND_NUDGE |
| Task 2 — Schedule Optimisation | ~0.45 | Agent struggles with stress constraints |
| Task 3 — Distraction Mitigation | ~0.35 | Hard task; distractions accumulate quickly |

---

## Environment Architecture

```
Productivity_Copilot/
├── productivity_env/       # Core OpenEnv environment package
│   ├── env.py              # ProductivityEnv class (step, reset)
│   ├── models.py           # Pydantic Observation & Action models
│   └── __init__.py
├── data_pipeline/          # ML model loading + inference helpers
│   └── inference.py        # CopilotModels singleton (loads .pkl files)
├── model_artifacts/        # Trained .pkl model files
│   ├── failure_predictor.pkl
│   ├── distraction_scorer.pkl
│   └── work_style_classifier.pkl
├── vectorstore/            # ChromaDB RAG coaching knowledge base
├── server/
│   └── app.py              # FastAPI server for HF Space
├── inference.py            # Baseline agent evaluation script
├── openenv.yaml            # OpenEnv metadata manifest
├── pyproject.toml          # Python project config
├── uv.lock                 # Locked dependencies
└── Dockerfile              # HuggingFace Space container
```
