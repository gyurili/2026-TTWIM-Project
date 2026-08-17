from fastapi import APIRouter, Depends
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.common import SuccessResponse
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse
from app.services import predict as service
from app.services.coaching import STABLE_TAG, classify_slope, generate_coaching, get_stable_coaching_text

router = APIRouter(tags=["Predict"])

_CHANNEL_FIELDS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _latest_abnormal_run_indices(predictions) -> list[int]:
    """가장 최근 비정상 윈도우와 그 앞의 연속된 비정상 묶음 인덱스를 반환한다.

    중간의 정상 구간을 제거하고 떨어진 비정상 윈도우를 이어 붙이면 걸음 간격과
    리듬 지표가 왜곡될 수 있다. 마지막 윈도우가 정상이더라도, 그 직전의 가장
    최근 비정상 묶음은 이번 배치의 코칭 근거로 유지한다.
    """
    latest_abnormal_index = next(
        (index for index in range(len(predictions) - 1, -1, -1) if predictions[index].prediction_code == 1),
        None,
    )
    if latest_abnormal_index is None:
        return []

    first_index = latest_abnormal_index
    while first_index > 0 and predictions[first_index - 1].prediction_code == 1:
        first_index -= 1
    return list(range(first_index, latest_abnormal_index + 1))


@router.post("/predict", response_model=SuccessResponse[PredictBatchResponse])
def predict_heel_strike(data: PredictBatchRequest, current_user: User = Depends(get_current_user)):
    result = service.predict_batch(data)

    altitude_readings = data.altitudeContext.readings if data.altitudeContext else None

    abnormal_indices = [i for i, p in enumerate(result.predictions) if p.prediction_code == 1]
    normal_count = len(result.predictions) - len(abnormal_indices)
    if len(abnormal_indices) > normal_count:
        # 전체 비정상 비율은 기존처럼 배치 전체 기준으로 쓰되, 센서 지표는 시간적으로
        # 이어진 가장 최근 비정상 구간만으로 계산한다.
        recent_abnormal_indices = _latest_abnormal_run_indices(result.predictions)
        abnormal_windows = [data.steps[i] for i in recent_abnormal_indices]
        channels = {
            field: [v for w in abnormal_windows for v in getattr(w, field)]
            for field in _CHANNEL_FIELDS
        }

        ratio = len(abnormal_indices) / len(data.steps)
        dtw = data.dtwContext.model_dump()
        coaching_result = generate_coaching(channels, ratio, dtw, altitude_readings=altitude_readings)

        if coaching_result is not None:
            text, tags, slope = coaching_result
            result.coaching_text = text
            result.coaching_tags = tags
            result.terrain = slope
    else:
        result.coaching_text = get_stable_coaching_text()
        result.coaching_tags = [STABLE_TAG]
        result.terrain = classify_slope(altitude_readings) if altitude_readings else None

    return SuccessResponse(data=result)
