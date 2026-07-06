from sqlalchemy import Column, BigInteger, Integer, String, Text, DateTime, Numeric, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.db.base import Base


class RunningSession(Base):
    __tablename__ = "running_sessions"

    session_id         = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id            = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    start_location     = Column(JSONB)
    end_location       = Column(JSONB)
    start_time         = Column(DateTime, nullable=False)
    end_time           = Column(DateTime, nullable=False)
    total_distance_m   = Column(Numeric(10, 2), nullable=False)
    total_duration_sec = Column(Integer, nullable=False)
    created_at         = Column(DateTime, server_default=func.now())


class SessionWindow(Base):
    __tablename__ = "session_windows"

    window_id        = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id       = Column(BigInteger, ForeignKey("running_sessions.session_id", ondelete="CASCADE"), nullable=False)
    prediction       = Column(String(20), nullable=False)
    tag              = Column(String(50))
    feedback_text    = Column(String(2000))
    dtw_context      = Column(JSONB)
    window_start_time = Column(DateTime, nullable=False)
    window_end_time   = Column(DateTime, nullable=False)
    gps_start        = Column(JSONB)
    gps_end          = Column(JSONB)


class RunningFeedback(Base):
    __tablename__ = "running_feedback"

    feedback_id    = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id     = Column(BigInteger, ForeignKey("running_sessions.session_id", ondelete="CASCADE"), nullable=False)
    user_id        = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    final_feedback = Column(Text, nullable=False)
    tag_ratio      = Column(JSONB, nullable=False)
    created_at     = Column(DateTime, server_default=func.now())
