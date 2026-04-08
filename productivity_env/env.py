from typing import Any, Dict, Optional

import os
import sys

from openenv.core import Environment

from .models import ProductivityAction, ProductivityObservation, ProductivityState

# Add parent directory to path so data_pipeline can be imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from data_pipeline.inference import copilot


class ProductivityEnv(Environment[ProductivityAction, ProductivityObservation, ProductivityState]):
    def __init__(self, task_name: str = "triage"):
        super().__init__()
        self.task_name = task_name
        self.state_data: Dict[str, Any] = {}
        self.max_steps = 10
        self.current_step = 0
        self.episode_id: Optional[str] = None
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
        self.current_step = 0
        self.episode_id = episode_id
        task_name = kwargs.get("task_name", self.task_name)
        if task_name:
            self.task_name = str(task_name)

        if self.task_name == "triage":
            self.state_data = {
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
            }
        elif self.task_name == "schedule_optimization":
            self.state_data = {
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
            }
        elif self.task_name == "distraction_mitigation":
            self.state_data = {
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
            }
        else:
            self.state_data = {
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

        obs = self._get_obs()
        obs.reward = 0.0
        obs.done = False
        obs.metadata = {"task_name": self.task_name, "episode_id": self.episode_id, "seed": seed}
        return obs

    def _get_obs(self) -> ProductivityObservation:
        if not self.state_data:
            self.reset()

        fp_res = copilot.predict_failure(self.state_data)
        dist_res = copilot.score_distraction(self.state_data)

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
        reward = (1.0 - obs.failure_probability) * 0.1
        if obs.stress_level >= 8.0:
            reward -= 0.05

        obs.reward = reward
        obs.done = self.current_step >= self.max_steps
        obs.metadata = {
            "task_name": self.task_name,
            "step_count": self.current_step,
            "timeout_s": timeout_s,
        }
        return obs

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
