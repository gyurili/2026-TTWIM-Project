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


class PredictBatchRequest(BaseModel):
    steps: list[StepFeatures]
    dtwContext: DtwContext


class StepPrediction(BaseModel):
    prediction_code: int
    prediction_label: str


class PredictBatchResponse(BaseModel):
    predictions: list[StepPrediction]
    coaching_text: str = ""
    coaching_tags: list[str] = []
