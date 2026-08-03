from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.exceptions import SessionForbiddenError, SessionNotFoundError
from app.core.security import get_current_user
from app.core.time import KST
from app.crud.running import create_session, get_session, get_weekly_distance_by_date
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import SuccessResponse
from app.schemas.running import (
    DailyDistanceOut,
    HomeSummaryOut,
    RunningFeedbackOut,
    RunningSessionCreate,
    RunningSessionOut,
)
from app.services.feedback import generate_session_feedback

router = APIRouter(tags=["Running"])

WEEKDAY_LABELS = ["월", "화", "수", "목", "금", "토", "일"]


@router.get("/running/home", response_model=SuccessResponse[HomeSummaryOut])
def get_home_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    today = datetime.now(KST).date()
    week_start = today - timedelta(days=today.weekday())
    distance_by_date = get_weekly_distance_by_date(db, current_user.id, week_start, week_start + timedelta(days=7))

    daily_distances = []
    week_total_m = Decimal("0")
    for i, label in enumerate(WEEKDAY_LABELS):
        day = week_start + timedelta(days=i)
        day_total_m = distance_by_date.get(day, Decimal("0"))
        week_total_m += day_total_m
        daily_distances.append(
            DailyDistanceOut(date=day, weekday_label=label, distance_km=round(float(day_total_m) / 1000, 1))
        )

    result = HomeSummaryOut(
        nickname=current_user.nickname,
        week_total_km=round(float(week_total_m) / 1000, 1),
        daily_distances=daily_distances,
    )
    return SuccessResponse(data=result)


@router.post("/running/complete", response_model=SuccessResponse[RunningSessionOut])
def complete_running(
    data: RunningSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = create_session(db, current_user.id, data)
    return SuccessResponse(data=session)


@router.post("/running/{session_id}/feedback", response_model=SuccessResponse[RunningFeedbackOut])
def create_session_feedback(
    session_id: int,
    force_regenerate: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = get_session(db, session_id)
    if not session:
        raise SessionNotFoundError()
    if session.user_id != current_user.id:
        raise SessionForbiddenError()
    result = generate_session_feedback(db, session, current_user.id, current_user, force_regenerate)
    return SuccessResponse(data=result)
