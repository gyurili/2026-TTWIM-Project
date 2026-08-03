import uuid
from sqlalchemy import Boolean, Column, String, Integer, Date, DateTime, text
from sqlalchemy.dialects.postgresql import UUID

from app.core.time import now_kst_naive
from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(50), unique=True, nullable=False)
    nickname = Column(String(50), nullable=False)
    password_hash = Column(String(128), nullable=False)
    gender = Column(String(6), nullable=False)  # 'man' or 'woman'
    birth_date = Column(Date, nullable=False)    # YYYY-MM-DD 저장
    height = Column(Integer, nullable=False)     # cm
    weight = Column(Integer, nullable=False)     # kg
    is_calibrated = Column(  # DTW 캘리브레이션 완료 여부
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    created_at = Column(DateTime, default=now_kst_naive)
