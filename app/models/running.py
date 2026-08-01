from sqlalchemy import Column, BigInteger, Integer, String, Text, DateTime, Numeric, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.db.base import Base


class RunningSession(Base):
    __tablename__ = "running_sessions"

    session_id         = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id            = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    mode               = Column(String(10), nullable=False)
    route              = Column(JSONB)
    start_time         = Column(DateTime, nullable=False)
    end_time           = Column(DateTime, nullable=False)
    total_distance_m   = Column(Numeric(10, 2), nullable=False)
    total_duration_sec = Column(Integer, nullable=False)
    created_at         = Column(DateTime, server_default=func.now())

    windows = relationship("SessionWindow", backref="session", cascade="all, delete-orphan")


class SessionWindow(Base):
    __tablename__ = "session_windows"

    window_id        = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id       = Column(BigInteger, ForeignKey("running_sessions.session_id", ondelete="CASCADE"), nullable=False)
    predictions      = Column(JSONB, nullable=False)   # 윈도우 5개 예측 결과 배열
    tags             = Column(JSONB)                   # 코칭 태그 배열
    feedback_text    = Column(String(2000))
    dtw_context      = Column(JSONB)
    window_start_time = Column(BigInteger, nullable=False)  # epoch ms, route[].timestamp와 동일 기준
    window_end_time   = Column(BigInteger, nullable=False)


class RunningFeedback(Base):
    __tablename__ = "running_feedback"

    feedback_id    = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id     = Column(BigInteger, ForeignKey("running_sessions.session_id", ondelete="CASCADE"), nullable=False, unique=True)
    user_id        = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    final_feedback = Column(Text, nullable=False)
    tag_ratio      = Column(JSONB, nullable=False)
    timeline       = Column(JSONB, nullable=True)  # 세션 시작 기준 분 단위 구간별 우세 태그. window 재조회 비용 때문에 생성 시점에 캐싱.
    created_at     = Column(DateTime, server_default=func.now())
