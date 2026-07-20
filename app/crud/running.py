from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.running import RunningSession
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
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session
