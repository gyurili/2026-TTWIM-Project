from sqlalchemy import Column, BigInteger, String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.time import now_kst_naive
from app.db.base import Base


class DtwCalibration(Base):

    __tablename__ = "dtw_calibrations"

    dtw_id             = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id            = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    axis_name          = Column(String(20), nullable=False)   # 기준 축 (예: 'GyroZ')
    template           = Column(JSONB, nullable=False)        # 캘리브레이션 템플릿 배열
    base_dist          = Column(Float, nullable=False)        # 템플릿과의 거리 중앙값
    dist_mad           = Column(Float, nullable=False)        # 거리의 산포도(MAD)
    baseline_cadence   = Column(Float, nullable=False)        # 기준 케이던스
    cadence_mad        = Column(Float, nullable=False)        # 케이던스의 산포도(MAD)
    baseline_amplitude = Column(Float, nullable=False)        # 기준 진폭
    amplitude_mad      = Column(Float, nullable=False)        # 진폭의 산포도(MAD)
    created_at         = Column(DateTime, default=now_kst_naive)
    updated_at         = Column(DateTime, default=now_kst_naive, onupdate=now_kst_naive)
