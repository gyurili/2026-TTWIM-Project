from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.running import RunningSession, SessionWindow
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
    db.commit()  # 세션 + 윈도우 한 트랜잭션으로 저장 (relationship cascade)
    db.refresh(db_session)
    return db_session
