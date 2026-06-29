from typing import Optional

from pydantic import BaseModel


class StepFeatures(BaseModel):
    acc_x: list[float]
    acc_y: list[float]
    acc_z: list[float]
    gyro_x: list[float]
    gyro_y: list[float]
    gyro_z: list[float]


class PredictBatchRequest(BaseModel):
    steps: list[StepFeatures]


class StepPrediction(BaseModel):
    prediction_code: int
    prediction_label: str


class SensorStats(BaseModel):
    acc_z_mean: float
    acc_z_max: float
    gyro_mag_peak: float
    impact_score: float
    step_count: int
    stride_regularity: float


class PredictBatchResponse(BaseModel):
    status: str
    predictions: list[StepPrediction]
    needs_coaching: bool = False
    coaching_text: str = ""
    coaching_type: str = ""
    coaching_type_label: str = ""
    stats: Optional[SensorStats] = None
