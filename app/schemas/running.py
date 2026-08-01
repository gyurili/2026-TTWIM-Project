import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import field_validator

from app.schemas.base import CamelModel


class RoutePoint(CamelModel):
    latitude: float
    longitude: float
    timestamp: int


class SessionWindowCreate(CamelModel):
    predictions: list[str]
    tags: list[str] = []
    feedback_text: str | None = None
    dtw_context: dict[str, Any] | None = None
    window_start_time: int  # epoch ms, route[].timestamp와 동일 기준
    window_end_time: int


class RunningSessionCreate(CamelModel):
    mode: Literal["outdoor", "indoor"]
    route: list[RoutePoint] | None = None
    started_at: int
    ended_at: int
    duration_sec: int
    distance_meters: float
    windows: list[SessionWindowCreate] = []


class SessionWindowOut(CamelModel):
    window_id: int
    session_id: int
    predictions: list[str]
    tags: list[str] = []
    feedback_text: str | None = None
    dtw_context: dict[str, Any] | None = None
    window_start_time: int
    window_end_time: int


class RunningSessionOut(CamelModel):
    session_id: int
    user_id: UUID
    mode: str
    route: list[dict[str, Any]] | None = None
    start_time: datetime
    end_time: datetime
    total_distance_m: Decimal
    total_duration_sec: int
    created_at: datetime | None = None
    windows: list[SessionWindowOut] = []


class FeedbackDetail(CamelModel):
    summary: str
    strengths: list[str] = []
    improvements: list[str] = []
    next_step: str = ""


class RunningFeedbackOut(CamelModel):
    feedback_id: int
    session_id: int
    user_id: UUID
    final_feedback: FeedbackDetail
    tag_ratio: dict[str, Any]
    created_at: datetime | None = None

    @field_validator("final_feedback", mode="before")
    @classmethod
    def _parse_final_feedback(cls, v: Any) -> Any:
        """DB에는 JSON 문자열로 저장되어 있다. 이 스키마 도입 이전에 생성된 row는
        평문 텍스트라 JSON 파싱이 안 되므로, 그 경우 summary 한 줄짜리 결과로 감싼다."""
        if not isinstance(v, str):
            return v
        try:
            parsed = json.loads(v)
        except json.JSONDecodeError:
            return {"summary": v}
        return parsed if isinstance(parsed, dict) else {"summary": v}
