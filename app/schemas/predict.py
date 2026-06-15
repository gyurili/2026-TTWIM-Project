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


class PredictBatchResponse(BaseModel):
    status: str
    predictions: list[StepPrediction]
