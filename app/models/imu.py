from sqlalchemy import Column, BigInteger, Float, Integer, String, DateTime, Text, func, ForeignKey, Index
from app.db.base import Base


class GaitSession(Base):
    __tablename__ = "gait_sessions"

    id         = Column(String(50), primary_key=True, index=True)
    label      = Column(String(50), nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class GaitRecord(Base):
    __tablename__ = "gait_records"

    __table_args__ = (
        Index('idx_gait_session_window', 'session_id', 'window_index'),
    )

    id           = Column(BigInteger, primary_key=True, index=True)
    session_id   = Column(String(50), ForeignKey("gait_sessions.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    window_index = Column(Integer,    nullable=False)
    start_time   = Column(DateTime,   nullable=False)
    end_time     = Column(DateTime,   nullable=False)
    label        = Column(String(50), default='normal')
    subject      = Column(String(100), default='')

    raw_acc_x  = Column(Text, nullable=False)
    raw_acc_y  = Column(Text, nullable=False)
    raw_acc_z  = Column(Text, nullable=False)
    raw_gyro_x = Column(Text, nullable=False)
    raw_gyro_y = Column(Text, nullable=False)
    raw_gyro_z = Column(Text, nullable=False)

    received_at = Column(DateTime, server_default=func.now())
