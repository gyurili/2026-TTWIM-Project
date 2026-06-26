from pydantic import BaseModel, Field, field_validator


class FootDragStepRequest(BaseModel):
    acc_x: list[float] = Field(..., min_length=10)
    acc_y: list[float] = Field(..., min_length=10)
    acc_z: list[float] = Field(..., min_length=10)
    contact_time: float = Field(..., gt=0)
    gyro_x: list[float] | None = None
    gyro_y: list[float] | None = None
    gyro_z: list[float] | None = None

    @field_validator("acc_x", "acc_y", "acc_z")
    @classmethod
    def validate_acc_length(cls, value: list[float]) -> list[float]:
        if len(value) < 10:
            raise ValueError("가속도 시계열은 최소 10개 이상이어야 합니다.")
        return value

    def lengths_match(self) -> bool:
        return len(self.acc_x) == len(self.acc_y) == len(self.acc_z)


class FootDragBatchRequest(BaseModel):
    steps: list[FootDragStepRequest] = Field(..., min_length=1)


class FootDragPredictionResponse(BaseModel):
    prediction_code: int
    prediction_label: str
    confidence: float
    is_foot_drag: bool
    message: str
    dtw_ratio: float
    dtw_threshold: float
    model_threshold: float


class FootDragBatchResponse(BaseModel):
    status: str
    predictions: list[FootDragPredictionResponse]
