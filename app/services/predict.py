import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse, StepPrediction

MODEL_PATH        = os.path.join(os.path.dirname(__file__), "..", "data", "ensemble_model_gr.pkl")
FEAT_NAMES_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "feature_names_gr.pkl")

PREDICTION_LABELS = {
    0: "정상 보행",
    1: "힐스트라이크",
}

try:
    _model        = joblib.load(MODEL_PATH)
    _feat_names   = joblib.load(FEAT_NAMES_PATH)
    print(f"[predict service] 모델 로드 성공 | 특징: {_feat_names}")
except Exception as e:
    _model      = None
    _feat_names = None
    print(f"[predict service] 모델 로드 실패: {e}")


def _butter_low(data, cutoff=40.0, fs=100.0, order=4):
    b, a = butter(order, cutoff / (fs * 0.5), btype="low", analog=False)
    x = np.array(data, dtype=float)
    padlen = min(len(b) * 3, len(x) - 1)
    return filtfilt(b, a, x, padlen=padlen)


def _extract_features(step) -> dict:
    ax = _butter_low(step.acc_x)
    ay = _butter_low(step.acc_y)
    az = _butter_low(step.acc_z)
    gx = _butter_low(step.gyro_x)
    gy = _butter_low(step.gyro_y)
    gz = _butter_low(step.gyro_z)

    mag      = np.sqrt(ax**2 + ay**2 + az**2)
    peak_idx = int(np.argmax(az))

    return {
        # 자이로 특징 (판별력 1~4위)
        'gx_max'          : float(np.max(np.abs(gx))),
        'gy_max'          : float(np.max(np.abs(gy))),
        'gz_max'          : float(np.max(np.abs(gz))),
        'gyro_energy'     : float(np.sum(gx**2 + gy**2 + gz**2) / 100),
        'gy_mean'         : float(np.mean(np.abs(gy))),

        # 가속도 특징 (보조)
        'az_max'          : float(np.max(az)),
        'az_mean'         : float(np.mean(az)),
        'az_std'          : float(np.std(az)),
        'mag_max'         : float(np.max(mag)),
        'mag_std'         : float(np.std(mag)),

        # 충격 타이밍
        'peak_idx'        : peak_idx,
        'rise_rate'       : float(az[peak_idx] / (peak_idx + 1)),

        # 전반부/후반부 에너지 비
        'front_back_ratio': float(np.sum(az[:50]**2) / (np.sum(az[50:]**2) + 1e-6)),

        # 자이로 회전 최대 시점
        'gy_peak_idx'     : int(np.argmax(np.abs(gy))),

        # 충격 지속폭 (반값 너비)
        'impact_width'    : int(np.sum(az > az.max() * 0.5)),
    }


def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    if _model is None or _feat_names is None:
        raise RuntimeError("모델 또는 특징 이름 파일이 로드되지 않았습니다.")

    if not request.steps:
        return PredictBatchResponse(status="success", predictions=[])

    rows  = [_extract_features(step) for step in request.steps]
    X     = pd.DataFrame(rows, columns=_feat_names)
    preds = _model.predict(X)

    predictions = [
        StepPrediction(
            prediction_code=int(p),
            prediction_label=PREDICTION_LABELS.get(int(p), "알 수 없음"),
        )
        for p in preds
    ]

    return PredictBatchResponse(status="success", predictions=predictions)