from fastapi import APIRouter
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse, SensorStats
from app.services import predict as service
from app.services.coaching import extract_stats, generate_coaching, COACHING_TYPES

router = APIRouter(tags=["Predict"])

_CHANNEL_FIELDS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


@router.post("/predict", response_model=PredictBatchResponse)
def predict_heel_strike(data: PredictBatchRequest, analyze: bool = False):
    result = service.predict_batch(data)

    if not analyze:
        return result

    abnormal_indices = [i for i, p in enumerate(result.predictions) if p.prediction_code == 1]
    normal_count = len(result.predictions) - len(abnormal_indices)
    if len(abnormal_indices) <= normal_count:
        result.needs_coaching = False
        return result

    abnormal_windows = [data.steps[i] for i in abnormal_indices]
    merged_window = [
        [v for w in abnormal_windows for v in getattr(w, field)]
        for field in _CHANNEL_FIELDS
    ]
    stats = extract_stats(merged_window)

    ratio = len(abnormal_indices) / len(data.steps)
    coaching_result = generate_coaching(stats, ratio)

    if coaching_result is not None:
        text, ctype = coaching_result
        result.needs_coaching = True
        result.coaching_text = text
        result.coaching_type = ctype
        result.coaching_type_label = COACHING_TYPES.get(ctype, ctype)
        result.stats = SensorStats(**stats)
    else:
        result.needs_coaching = False

    return result
