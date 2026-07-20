from fastapi import APIRouter, Depends
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.common import SuccessResponse
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse
from app.services import predict as service
from app.services.coaching import generate_coaching_raw

router = APIRouter(tags=["Predict"])

_CHANNEL_FIELDS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


@router.post("/predict", response_model=SuccessResponse[PredictBatchResponse])
def predict_heel_strike(data: PredictBatchRequest, current_user: User = Depends(get_current_user)):
    result = service.predict_batch(data)

    abnormal_indices = [i for i, p in enumerate(result.predictions) if p.prediction_code == 1]
    normal_count = len(result.predictions) - len(abnormal_indices)
    if len(abnormal_indices) > normal_count:
        abnormal_windows = [data.steps[i] for i in abnormal_indices]
        channels = {
            field: [v for w in abnormal_windows for v in getattr(w, field)]
            for field in _CHANNEL_FIELDS
        }

        ratio = len(abnormal_indices) / len(data.steps)
        dtw = data.dtwContext.model_dump()
        coaching_result = generate_coaching_raw(channels, ratio, dtw)

        if coaching_result is not None:
            text, tags = coaching_result
            result.coaching_text = text
            result.coaching_tags = tags

    return SuccessResponse(data=result)
