from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.running import RunningFeedback, RunningSession, SessionWindow
from app.schemas.running import RunningSessionCreate


def create_session(db: Session, user_id: UUID, data: RunningSessionCreate) -> RunningSession:
    db_session = RunningSession(
        user_id=user_id,
        mode=data.mode,
        route=[r.model_dump() for r in data.route] if data.route else None,
        start_time=datetime.fromtimestamp(data.started_at / 1000),
        end_time=datetime.fromtimestamp(data.ended_at / 1000),
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
    else:
        existing = RunningFeedback(session_id=session_id, user_id=user_id, **data)
        db.add(existing)
    db.commit()
    db.refresh(existing)
    return existing


def get_windows_by_session(db: Session, session_id: int) -> list[SessionWindow]:
    return (
        db.query(SessionWindow)
        .filter(SessionWindow.session_id == session_id)
        .order_by(SessionWindow.window_start_time)
        .all()
    )
