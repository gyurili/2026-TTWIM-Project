from sqlalchemy import Column, Integer, Float, String, DateTime, func
from app.db.base import Base


class IMURecord(Base):
    __tablename__ = "imu_records"

    id = Column(Integer, primary_key=True, index=True)
    side = Column(String(10), nullable=False)  # "left" / "right"

    # 가속도계
    acc_x = Column(Float, nullable=False)
    acc_y = Column(Float, nullable=False)
    acc_z = Column(Float, nullable=False)

    # 자이로스코프
    gyro_x = Column(Float, nullable=False)
    gyro_y = Column(Float, nullable=False)
    gyro_z = Column(Float, nullable=False)

    # RN 앱에서 기록한 시각 (한국 시간)
    created_at = Column(DateTime, nullable=False)

    # 서버 수신 시각 (자동)
    received_at = Column(DateTime, server_default=func.now())

    # 데이터 수집용 추가 정보
    user_id    = Column(String(50), nullable=True)
    session_id = Column(String(50), nullable=True)
    label      = Column(String(50), nullable=True)  # "normal", "heel_strike", "overpronation" 등
    weight     = Column(Float, nullable=True)        # 체중 (kg)
    height     = Column(Float, nullable=True)        # 키 (cm)