from calendar import monthrange
from datetime import date, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.time import epoch_ms_to_kst_naive
from app.models.running import RunningFeedback, RunningSession, SessionWindow
from app.schemas.running import RunningSessionCreate


def create_session(db: Session, user_id: UUID, data: RunningSessionCreate) -> RunningSession:
    db_session = RunningSession(
        user_id=user_id,
        mode=data.mode,
        route=[r.model_dump() for r in data.route] if data.route else None,
        start_time=epoch_ms_to_kst_naive(data.started_at),
        end_time=epoch_ms_to_kst_naive(data.ended_at),
        total_distance_m=Decimal(str(round(data.distance_meters, 2))),
        total_duration_sec=data.duration_sec,
    )
    db_session.windows = [
        SessionWindow(
            predictions=w.predictions,
            tags=w.tags,
            feedback_text=w.feedback_text,
            dtw_context=w.dtw_context,
            window_start_time=w.window_start_time,
            window_end_time=w.window_end_time,
        )
        for w in data.windows
    ]

    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def get_session(db: Session, session_id: int) -> RunningSession | None:
    return db.query(RunningSession).filter(RunningSession.session_id == session_id).first()


def get_feedback_by_session(db: Session, session_id: int) -> RunningFeedback | None:
    return db.query(RunningFeedback).filter(RunningFeedback.session_id == session_id).first()


def upsert_feedback(db: Session, session_id: int, user_id: UUID, data: dict) -> RunningFeedback:
    existing = get_feedback_by_session(db, session_id)
    if existing:
        existing.final_feedback = data["final_feedback"]
        existing.tag_ratio = data["tag_ratio"]
        existing.timeline = data["timeline"]
    else:
        existing = RunningFeedback(session_id=session_id, user_id=user_id, **data)
        db.add(existing)
    db.commit()
    db.refresh(existing)
    return existing


def get_total_distance_m(db: Session, user_id: UUID) -> Decimal:
    """해당 사용자의 누적 총 러닝 거리(m)를 반환한다. 러닝 기록이 없으면 0."""
    total = (
        db.query(func.sum(RunningSession.total_distance_m))
        .filter(RunningSession.user_id == user_id)
        .scalar()
    )
    return total or Decimal("0")


def get_weekly_distance_by_date(
    db: Session, user_id: UUID, week_start: date, week_end_exclusive: date
) -> dict[date, Decimal]:
    rows = (
        db.query(func.date(RunningSession.start_time), func.sum(RunningSession.total_distance_m))
        .filter(RunningSession.user_id == user_id)
        .filter(RunningSession.start_time >= week_start, RunningSession.start_time < week_end_exclusive)
        .group_by(func.date(RunningSession.start_time))
        .all()
    )
    return dict(rows)


def get_monthly_sessions_with_feedback(
    db: Session, user_id: UUID, year: int, month: int
) -> list[tuple[RunningSession, RunningFeedback | None]]:
    start = date(year, month, 1)
    end_exclusive = date(year, month, monthrange(year, month)[1]) + timedelta(days=1)
    return (
        db.query(RunningSession, RunningFeedback)
        .outerjoin(RunningFeedback, RunningFeedback.session_id == RunningSession.session_id)
        .filter(RunningSession.user_id == user_id)
        .filter(RunningSession.start_time >= start, RunningSession.start_time < end_exclusive)
        .order_by(RunningSession.start_time.desc())
        .all()
    )


def get_windows_by_session(db: Session, session_id: int) -> list[SessionWindow]:
    return (
        db.query(SessionWindow)
        .filter(SessionWindow.session_id == session_id)
        .order_by(SessionWindow.window_start_time)
        .all()
    )
