from pydantic import BaseModel, Field
from datetime import datetime
from typing import List


class IMURecordCreate(BaseModel):
    side: str
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    created_at: datetime



class IMUBatchUpload(BaseModel):
    records: List[IMURecordCreate] = Field(..., min_length=1)


class IMUBatchResponse(BaseModel):
    message: str
    count: int