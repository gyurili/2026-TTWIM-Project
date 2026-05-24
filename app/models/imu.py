from sqlalchemy import Column, BigInteger, Float, Integer, String, DateTime, Text, func, ForeignKey, Index
from app.db.base import Base


class GaitSession(Base):
    __tablename__ = "gait_sessions"

    id         = Column(String(50), primary_key=True, index=True)
    user_id    = Column(String(50), nullable=True, index=True)
    label      = Column(String(50), nullable=True)
    weight     = Column(Float, nullable=True)
    height     = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class GaitRecord(Base):
    __tablename__ = "gait_records"

    __table_args__ = (
        Index('idx_gait_session_step', 'session_id', 'step_index'),
    )

    id         = Column(BigInteger, primary_key=True, index=True)
    session_id = Column(String(50), ForeignKey("gait_sessions.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    side       = Column(String(10), nullable=False)
    step_index = Column(Integer,    nullable=False)
    start_time = Column(DateTime,   nullable=False)
    end_time   = Column(DateTime,   nullable=False)

    peak_acc_z   = Column(Float,      nullable=False)
    cadence      = Column(Float,      nullable=False)
    contact_time = Column(Float,      nullable=False)
    gait_label   = Column(String(50), default='normal')

    resampled_acc_x  = Column(Text, nullable=False)
    resampled_acc_y  = Column(Text, nullable=False)
    resampled_acc_z  = Column(Text, nullable=False)
    resampled_gyro_x = Column(Text, nullable=False)
    resampled_gyro_y = Column(Text, nullable=False)
    resampled_gyro_z = Column(Text, nullable=False)

    received_at = Column(DateTime, server_default=func.now())