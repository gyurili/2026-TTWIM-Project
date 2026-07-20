from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RoutePoint(BaseModel):
    latitude: float
    longitude: float
    timestamp: int


class RunningSessionCreate(BaseModel):
    mode: Literal["outdoor", "indoor"]
    route: list[RoutePoint] | None = None
    started_at: int
    ended_at: int
    duration_sec: Annotated[int, Field(alias="durationSec")]
    distance_meters: Annotated[float, Field(alias="distanceMeters")]


class RunningSessionOut(BaseModel):
    session_id: int
    user_id: UUID
    mode: str
    route: list[dict[str, Any]] | None = None
    start_time: datetime
    end_time: datetime
    total_distance_m: Decimal
    total_duration_sec: int
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class SessionWindowCreate(BaseModel):
    prediction: str
    tag: str | None = None
    feedback_text: str | None = None
    dtw_context: dict[str, Any] | None = None
    window_start_time: datetime
    window_end_time: datetime
    gps_start: dict[str, Any] | None = None
    gps_end: dict[str, Any] | None = None


class SessionWindowOut(BaseModel):
    window_id: int
    session_id: int
    prediction: str
    tag: str | None = None
    feedback_text: str | None = None
    dtw_context: dict[str, Any] | None = None
    window_start_time: datetime
    window_end_time: datetime
    gps_start: dict[str, Any] | None = None
    gps_end: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class RunningFeedbackCreate(BaseModel):
    final_feedback: str
    tag_ratio: dict[str, Any]


class RunningFeedbackOut(BaseModel):
    feedback_id: int
    session_id: int
    user_id: UUID
    final_feedback: str
    tag_ratio: dict[str, Any]
    created_at: datetime | None = None

    model_config = {"from_attributes": True}
