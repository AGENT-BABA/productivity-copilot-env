from pydantic import BaseModel
from typing import Optional

class ProductivityObservation(BaseModel):
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

class ProductivityAction(BaseModel):
    action_type: str  # e.g., "WAIT", "SEND_NUDGE", "FORCE_BREAK", "BLOCK_SOCIAL_MEDIA"
    message: Optional[str] = None
