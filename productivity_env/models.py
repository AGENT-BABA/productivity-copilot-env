from typing import Any, Optional

from openenv.core import Action, Observation, State


class ProductivityObservation(Observation):
    time_of_day_hour: float
    stress_level: float
    distraction_events: int
    focus_score: float
    motivation_level: float
    session_duration_minutes: int
    break_count: int
    social_media_minutes: int
    current_task: str
    deadline_days_remaining: float
    failure_probability: float

class ProductivityAction(Action):
    action_type: str  # e.g., "WAIT", "SEND_NUDGE", "FORCE_BREAK", "BLOCK_SOCIAL_MEDIA"
    message: Optional[str] = None


class ProductivityState(State):
    task_name: str
    current_task: str
    deadline_days_remaining: float
    stress_level: float
    motivation_level: float
    distraction_events: int
    focus_score: float
    failure_probability: float
    session_duration_minutes: int
    break_count: int
    social_media_minutes: int
    time_of_day_hour: float
    raw_state: dict[str, Any]
