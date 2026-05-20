from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class IMURecordCreate(BaseModel):
    side: str
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    created_at: datetime

    # 데이터 수집용 추가 정보
    user_id:    Optional[str]   = None
    session_id: Optional[str]   = None
    label:      Optional[str]   = None
    weight:     Optional[float] = None
    height:     Optional[float] = None


class IMUBatchUpload(BaseModel):
    records: List[IMURecordCreate] = Field(..., min_length=1)


class IMUBatchResponse(BaseModel):
    message: str
    count: int