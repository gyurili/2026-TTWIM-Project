from sqlalchemy import Column, Integer, BigInteger, Float, String, DateTime, func, ForeignKey, Index
from app.db.base import Base

class IMUSession(Base):
    __tablename__ = "imu_sessions"

    id = Column(String(50), primary_key=True, index=True)  # uuid4 값이 들어옴
    user_id = Column(String(50), nullable=True, index=True)
    label = Column(String(50), nullable=True)   # "normal", "heel_strike" 등
    weight = Column(Float, nullable=True)        # 체중 (kg)
    height = Column(Float, nullable=True)        # 키 (cm)
    created_at = Column(DateTime, server_default=func.now())


class IMURecord(Base):
    __tablename__ = "imu_records"
    
    # 💡 [추가] 속도 최적화를 위한 복합 인덱스 설정
    __table_args__ = (
        Index('idx_session_created_at', 'session_id', 'created_at'),
    )

    # 💡 [변경] 대용량 데이터 폭발을 대비해 Integer -> BigInteger로 변경
    id = Column(BigInteger, primary_key=True, index=True)
    
    session_id = Column(String(50), ForeignKey("imu_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    side = Column(String(10), nullable=False)

    # 가속도계
    acc_x = Column(Float, nullable=False)
    acc_y = Column(Float, nullable=False)
    acc_z = Column(Float, nullable=False)

    # 자이로스코프
    gyro_x = Column(Float, nullable=False)
    gyro_y = Column(Float, nullable=False)
    gyro_z = Column(Float, nullable=False)

    # 시각 정보
    created_at = Column(DateTime, nullable=False)
    received_at = Column(DateTime, server_default=func.now())