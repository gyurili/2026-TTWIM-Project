import os
import numpy as np
import joblib
import pandas as pd
from scipy.signal import butter, filtfilt
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse, StepPrediction

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ensemble_model_gr.pkl")

PREDICTION_LABELS = {
    0: "정상 보행",
    1: "힐스트라이크",
}

_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
_FEATURE_NAMES = [f"{axis}_{i}" for axis in _AXES for i in range(100)]

try:
    _model = joblib.load(MODEL_PATH)
except Exception as e:
    _model = None
    print(f"[predict service] 모델 로드 실패: {e}")


def _butter_lowpass_filter(data, cutoff=40.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    x = np.array(data, dtype=float)
    ntaps = max(len(a), len(b))
    padlen = min(ntaps * 3, len(x) - 1)
    return filtfilt(b, a, x, padlen=padlen)


def _step_to_raw(step) -> np.ndarray:
    filtered = [_butter_lowpass_filter(getattr(step, ax)) for ax in _AXES]
    return np.concatenate(filtered)


def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    if _model is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")

    if not request.steps:
        return PredictBatchResponse(status="success", predictions=[])

    rows = [_step_to_raw(step) for step in request.steps]
    X = pd.DataFrame(rows, columns=_FEATURE_NAMES)
    preds = _model.predict(X)

    predictions = [
        StepPrediction(
            prediction_code=int(p),
            prediction_label=PREDICTION_LABELS.get(int(p), "알 수 없음"),
        )
        for p in preds
    ]

    return PredictBatchResponse(status="success", predictions=predictions)
