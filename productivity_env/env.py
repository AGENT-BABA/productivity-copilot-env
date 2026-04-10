from typing import Any, Dict, Optional

import os
import sys

from openenv.core import Environment
from openenv.core.env_server.types import EnvironmentMetadata
from openenv.core.rubrics import Rubric

from .models import ProductivityAction, ProductivityObservation, ProductivityState

# Add parent directory to path so data_pipeline can be imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from data_pipeline.inference import copilot

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "triage": {
        "session_duration_minutes": 180,
        "break_count": 0,
        "social_media_minutes_before": 45,
        "task_complexity": 4,
        "work_style_score": 0.5,
        "time_of_day_hour": 15,
        "day_of_week": 3,
        "stress_level": 7,
        "sleep_hours": 6,
        "distraction_events": 10,
        "deadline_days_remaining": 1.0,
        "previous_completion_rate": 0.6,
        "focus_score": 0.3,
        "motivation_level": 4,
        "study_hours_weekly": 15,
        "current_task": "Write Final Report",
    },
    "schedule_optimization": {
        "session_duration_minutes": 60,
        "break_count": 3,
        "social_media_minutes_before": 10,
        "task_complexity": 5,
        "work_style_score": 0.9,
        "time_of_day_hour": 10,
        "day_of_week": 1,
        "stress_level": 4,
        "sleep_hours": 8,
        "distraction_events": 2,
        "deadline_days_remaining": 0.5,
        "previous_completion_rate": 0.9,
        "focus_score": 0.8,
        "motivation_level": 6,
        "study_hours_weekly": 40,
        "current_task": "Code Review",
    },
    "distraction_mitigation": {
        "session_duration_minutes": 240,
        "break_count": 1,
        "social_media_minutes_before": 120,
        "task_complexity": 2,
        "work_style_score": 0.2,
        "time_of_day_hour": 18,
        "day_of_week": 5,
        "stress_level": 5,
        "sleep_hours": 4,
        "distraction_events": 20,
        "deadline_days_remaining": 2.0,
        "previous_completion_rate": 0.5,
        "focus_score": 0.2,
        "motivation_level": 2,
        "study_hours_weekly": 5,
        "current_task": "Update Documentation",
    },
}

DEFAULT_TASK_STATE: Dict[str, Any] = {
    "session_duration_minutes": 120,
    "break_count": 2,
    "social_media_minutes_before": 15,
    "task_complexity": 3,
    "work_style_score": 0.5,
    "time_of_day_hour": 10,
    "day_of_week": 1,
    "stress_level": 5,
    "sleep_hours": 7,
    "distraction_events": 5,
    "deadline_days_remaining": 3.0,
    "previous_completion_rate": 0.7,
    "focus_score": 0.6,
    "motivation_level": 6,
    "study_hours_weekly": 20,
    "current_task": "General Work",
}


def _clamp_task_score(score: float) -> float:
    return min(max(score, 0.01), 0.99)


def _compute_task_score(
    task_name: str,
    obs: ProductivityObservation,
    focus_history: list[float],
) -> float:
    if task_name == "triage":
        score = 0.15 + 0.7 * (1.0 - obs.failure_probability)
    elif task_name == "schedule_optimization":
        stress_bonus = max(0.0, 1.0 - max(obs.stress_level - 7.0, 0.0) / 3.0)
        score = 0.1 + 0.5 * (1.0 - obs.failure_probability) + 0.3 * stress_bonus
    elif task_name == "distraction_mitigation":
        avg_focus = sum(focus_history) / max(len(focus_history), 1)
        focus_bonus = max(0.0, 1.0 - avg_focus)
        score = 0.1 + 0.35 * (1.0 - obs.failure_probability) + 0.35 * focus_bonus
    else:
        score = 0.15 + 0.6 * (1.0 - obs.failure_probability)

    return _clamp_task_score(score)


class _TaskRubric(Rubric):
    def __init__(self, env: "ProductivityEnv", task_name: str):
        super().__init__()
        self.env = env
        self.task_name = task_name

    def forward(self, action: Any, observation: ProductivityObservation) -> float:
        return _compute_task_score(
            self.task_name,
            observation,
            self.env.focus_history,
        )


class ProductivityTaskRubric(Rubric):
    def __init__(self, env: "ProductivityEnv"):
        super().__init__()
        self.env = env
        for task_name in TASK_CONFIGS:
            setattr(self, task_name, _TaskRubric(env, task_name))

    def forward(self, action: Any, observation: ProductivityObservation) -> float:
        rubric = getattr(self, self.env.task_name, None)
        if isinstance(rubric, Rubric):
            return rubric(action, observation)
        return _compute_task_score(self.env.task_name, observation, self.env.focus_history)

    def sync_scores(self, observation: ProductivityObservation) -> dict[str, float]:
        scores: dict[str, float] = {}
        for task_name in TASK_CONFIGS:
            rubric = getattr(self, task_name)
            score = _compute_task_score(task_name, observation, self.env.focus_history)
            rubric.last_score = score
            scores[task_name] = score
        self.last_score = scores.get(self.env.task_name)
        return scores


class ProductivityEnv(Environment[ProductivityAction, ProductivityObservation, ProductivityState]):
    def __init__(self, task_name: str = "triage"):
        super().__init__(rubric=ProductivityTaskRubric(self))
        self.task_name = task_name
        self.state_data: Dict[str, Any] = {}
        self.max_steps = 10
        self.current_step = 0
        self.episode_id: Optional[str] = None
        self.focus_history: list[float] = []
        try:
            copilot.load()
        except Exception:
            print("Warning: Models could not be loaded. Please ensure model_artifacts are present.")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ProductivityObservation:
        self._reset_rubric()
        self.current_step = 0
        self.episode_id = episode_id
        self.focus_history = []
        task_name = kwargs.get("task_name", self.task_name)
        if task_name:
            self.task_name = str(task_name)

        self.state_data = dict(TASK_CONFIGS.get(self.task_name, DEFAULT_TASK_STATE))

        obs = self._get_obs()
        self.focus_history.append(obs.focus_score)
        rubric_scores = self.rubric.sync_scores(obs) if isinstance(self.rubric, ProductivityTaskRubric) else {}
        obs.reward = rubric_scores.get(self.task_name, _compute_task_score(self.task_name, obs, self.focus_history))
        obs.done = False
        obs.metadata = {
            "task_name": self.task_name,
            "episode_id": self.episode_id,
            "seed": seed,
            "available_tasks": list(TASK_CONFIGS.keys()),
            "task_scores": rubric_scores,
        }
        return obs

    def _get_obs(self) -> ProductivityObservation:
        if not self.state_data:
            self.reset()

        fp_res = self._safe_predict_failure()
        dist_res = self._safe_score_distraction()

        return ProductivityObservation(
            time_of_day_hour=float(self.state_data["time_of_day_hour"]),
            stress_level=float(max(0, min(10, self.state_data["stress_level"]))),
            distraction_events=int(max(0, self.state_data["distraction_events"])),
            focus_score=float(dist_res["distraction_score"]),
            motivation_level=float(max(0, min(10, self.state_data["motivation_level"]))),
            session_duration_minutes=int(self.state_data["session_duration_minutes"]),
            break_count=int(self.state_data["break_count"]),
            social_media_minutes=int(self.state_data["social_media_minutes_before"]),
            current_task=str(self.state_data["current_task"]),
            deadline_days_remaining=float(self.state_data["deadline_days_remaining"]),
            failure_probability=float(fp_res["failure_probability"]),
        )

    def _safe_predict_failure(self) -> Dict[str, Any]:
        try:
            return copilot.predict_failure(self.state_data)
        except Exception as exc:
            print(f"Warning: failure model unavailable, using heuristic score. Error: {exc}")
            stress = float(self.state_data.get("stress_level", 5.0)) / 10.0
            distractions = min(float(self.state_data.get("distraction_events", 0)) / 20.0, 1.0)
            deadline_pressure = max(0.0, 1.0 - min(float(self.state_data.get("deadline_days_remaining", 3.0)) / 3.0, 1.0))
            motivation = 1.0 - min(float(self.state_data.get("motivation_level", 5.0)) / 10.0, 1.0)
            risk_score = max(0.0, min(1.0, 0.35 * stress + 0.25 * distractions + 0.25 * deadline_pressure + 0.15 * motivation))
            return {
                "failure_probability": round(risk_score, 4),
                "risk_level": "high" if risk_score >= 0.65 else "medium" if risk_score >= 0.40 else "low",
                "should_intervene": risk_score >= 0.65,
            }

    def _safe_score_distraction(self) -> Dict[str, Any]:
        try:
            return copilot.score_distraction(self.state_data)
        except Exception as exc:
            print(f"Warning: distraction model unavailable, using heuristic score. Error: {exc}")
            distractions = min(float(self.state_data.get("distraction_events", 0)) / 20.0, 1.0)
            social = min(float(self.state_data.get("social_media_minutes_before", 0)) / 120.0, 1.0)
            focus = 1.0 - min(max(float(self.state_data.get("focus_score", 0.5)), 0.0), 1.0)
            score = max(0.0, min(1.0, 0.45 * distractions + 0.35 * social + 0.20 * focus))
            return {
                "distraction_score": round(score, 4),
                "level": "high" if score >= 0.65 else "medium" if score >= 0.35 else "low",
            }

    def step(
        self,
        action: ProductivityAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ProductivityObservation:
        self.current_step += 1

        self.state_data["session_duration_minutes"] += 30
        self.state_data["time_of_day_hour"] = (self.state_data["time_of_day_hour"] + 0.5) % 24
        self.state_data["deadline_days_remaining"] -= 30.0 / 1440.0
        self.state_data["distraction_events"] += 1

        action_type = action.action_type.upper()
        if action_type == "FORCE_BREAK":
            self.state_data["break_count"] += 1
            self.state_data["session_duration_minutes"] = 0
            self.state_data["stress_level"] = max(0, self.state_data["stress_level"] - 2)
            self.state_data["social_media_minutes_before"] += 5
        elif action_type == "BLOCK_SOCIAL_MEDIA":
            self.state_data["social_media_minutes_before"] = 0
            self.state_data["stress_level"] += 1
            self.state_data["distraction_events"] = max(0, self.state_data["distraction_events"] - 3)
        elif action_type == "SEND_NUDGE":
            self.state_data["motivation_level"] = min(10, self.state_data["motivation_level"] + 2)
            self.state_data["stress_level"] = max(0, self.state_data["stress_level"] - 0.5)
            self.state_data["distraction_events"] = max(0, self.state_data["distraction_events"] - 1)
        elif action_type == "WAIT" and self.state_data["stress_level"] > 6:
            self.state_data["stress_level"] += 0.5

        obs = self._get_obs()
        self.focus_history.append(obs.focus_score)
        rubric_scores = self.rubric.sync_scores(obs) if isinstance(self.rubric, ProductivityTaskRubric) else {}
        reward = self._apply_rubric(action, obs)

        obs.reward = reward
        obs.done = self.current_step >= self.max_steps
        obs.metadata = {
            "task_name": self.task_name,
            "step_count": self.current_step,
            "timeout_s": timeout_s,
            "available_tasks": list(TASK_CONFIGS.keys()),
            "task_scores": rubric_scores,
        }
        return obs

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="productivity-copilot-env",
            description=(
                "A productivity coaching environment with three graded tasks: "
                "triage, schedule_optimization, and distraction_mitigation."
            ),
            version="0.1.0",
            author="AGENT-BABA",
            documentation_url="https://github.com/AGENT-BABA/productivity-copilot-env",
        )

    @property
    def state(self) -> ProductivityState:
        obs = self._get_obs()
        return ProductivityState(
            episode_id=self.episode_id,
            step_count=self.current_step,
            task_name=self.task_name,
            current_task=obs.current_task,
            deadline_days_remaining=obs.deadline_days_remaining,
            stress_level=obs.stress_level,
            motivation_level=obs.motivation_level,
            distraction_events=obs.distraction_events,
            focus_score=obs.focus_score,
            failure_probability=obs.failure_probability,
            session_duration_minutes=obs.session_duration_minutes,
            break_count=obs.break_count,
            social_media_minutes=obs.social_media_minutes,
            time_of_day_hour=obs.time_of_day_hour,
            raw_state=dict(self.state_data),
        )
