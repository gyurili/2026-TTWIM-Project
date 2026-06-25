from pydantic import BaseModel, Field
from datetime import datetime
from typing import List


class GaitRecordCreate(BaseModel):
    window_index: int
    start_time  : datetime
    end_time    : datetime
    label       : str = 'normal'
    subject     : str = ''

    raw_acc_x : List[float]
    raw_acc_y : List[float]
    raw_acc_z : List[float]
    raw_gyro_x: List[float]
    raw_gyro_y: List[float]
    raw_gyro_z: List[float]


class GaitBatchUpload(BaseModel):
    records: List[GaitRecordCreate] = Field(..., min_length=1)


class GaitBatchResponse(BaseModel):
    message: str
    count  : int
