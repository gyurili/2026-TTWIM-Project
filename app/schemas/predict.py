from typing import Literal

from pydantic import BaseModel


class StepFeatures(BaseModel):
    acc_x: list[float]
    acc_y: list[float]
    acc_z: list[float]
    gyro_x: list[float]
    gyro_y: list[float]
    gyro_z: list[float]


class DtwContext(BaseModel):
    status: Literal["stable", "drift", "alert"]
    avgSimilarity: float
    trend: Literal["improving", "steady", "worsening"]
    shapeDev: float
    cadenceDev: float
    amplitudeDev: float
    timeline: list[float]


class AltitudeContext(BaseModel):
    readings: list[float]  # 윈도우 배치(약 20초)마다 1개씩 쌓이는 상대 고도(m), 최대 3개


class PredictBatchRequest(BaseModel):
    steps: list[StepFeatures]
    dtwContext: DtwContext
    altitudeContext: AltitudeContext | None = None


class StepPrediction(BaseModel):
    prediction_code: int
    prediction_label: str


class PredictBatchResponse(BaseModel):
    predictions: list[StepPrediction]
    coaching_tags: list[str] = []
    coaching_text: str = ""
    terrain: Literal["uphill", "downhill", "flat"] | None = None
