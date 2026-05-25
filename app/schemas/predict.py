from pydantic import BaseModel


class PredictRequest(BaseModel):
    acc_x: list[float]
    acc_y: list[float]
    acc_z: list[float]
    contact_time: float


class PredictResponse(BaseModel):
    status: str
    prediction_code: int
    prediction_label: str
