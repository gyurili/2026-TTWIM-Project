from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class GaitRecordCreate(BaseModel):
    side        : str
    step_index  : int
    start_time  : datetime
    end_time    : datetime
    peak_acc_z  : float
    cadence     : float
    contact_time: float
    label       : str = 'normal'

    resampled_acc_x : List[float]
    resampled_acc_y : List[float]
    resampled_acc_z : List[float]
    resampled_gyro_x: List[float]
    resampled_gyro_y: List[float]
    resampled_gyro_z: List[float]

    user_id    : Optional[str]   = None
    session_id : Optional[str]   = None
    weight     : Optional[float] = None
    height     : Optional[float] = None


class GaitBatchUpload(BaseModel):
    records: List[GaitRecordCreate] = Field(..., min_length=1)


class GaitBatchResponse(BaseModel):
    message: str
    count  : int