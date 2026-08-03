from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.exceptions import FeedbackNotFoundError, SessionForbiddenError, SessionNotFoundError
from app.core.security import get_current_user
from app.core.time import KST
from app.crud.running import (
    create_session,
    get_feedback_by_session,
    get_monthly_sessions_with_feedback,
    get_session,
    get_weekly_distance_by_date,
)
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import SuccessResponse
from app.schemas.running import (
    DailyDistanceOut,
    HomeSummaryOut,
    MonthlyRecentSessionOut,
    MonthlySummaryOut,
    MonthlyTagRatioOut,
    RunningFeedbackOut,
    RunningSessionCreate,
    RunningSessionOut,
    SessionFeedbackSummaryOut,
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


@router.get("/running/monthly-summary", response_model=SuccessResponse[MonthlySummaryOut])
def get_monthly_summary(
    year: int = Query(...),
    month: int = Query(..., ge=1, le=12),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = get_monthly_sessions_with_feedback(db, current_user.id, year, month)

    total_distance_m = sum((session.total_distance_m for session, _ in rows), Decimal("0"))
    tag_sums: dict[str, float] = {}
    feedback_count = 0
    recent_sessions = []

    for session, feedback in rows:
        dominant_tag = None
        if feedback and feedback.tag_ratio:
            feedback_count += 1
            for tag, ratio in feedback.tag_ratio.items():
                tag_sums[tag] = tag_sums.get(tag, 0) + ratio
            dominant_tag = max(feedback.tag_ratio.items(), key=lambda kv: kv[1])[0]
        recent_sessions.append(
            MonthlyRecentSessionOut(
                session_id=session.session_id,
                date=session.start_time.date(),
                distance_km=round(float(session.total_distance_m) / 1000, 2),
                dominant_tag=dominant_tag,
            )
        )

    tag_ratios = sorted(
        (MonthlyTagRatioOut(tag=tag, ratio=round(total / feedback_count, 2)) for tag, total in tag_sums.items()),
        key=lambda t: t.ratio,
        reverse=True,
    ) if feedback_count else []

    result = MonthlySummaryOut(
        year=year,
        month=month,
        total_distance_km=round(float(total_distance_m) / 1000, 2),
        total_count=len(rows),
        tag_ratios=tag_ratios,
        recent_sessions=recent_sessions,
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


@router.get("/running/{session_id}/feedback/summary", response_model=SuccessResponse[SessionFeedbackSummaryOut])
def get_session_feedback_summary(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = get_session(db, session_id)
    if not session:
        raise SessionNotFoundError()
    if session.user_id != current_user.id:
        raise SessionForbiddenError()

    feedback = get_feedback_by_session(db, session_id)
    if not feedback:
        raise FeedbackNotFoundError()

    return SuccessResponse(
        data=SessionFeedbackSummaryOut(tag_ratio=feedback.tag_ratio, timeline=feedback.timeline or [])
    )
