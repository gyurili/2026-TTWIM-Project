import os
import numpy as np
import onnxruntime as ort
from fastapi import HTTPException
from scipy.signal import butter, filtfilt
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse, StepPrediction

_CNN_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "cnn")
MODEL_PATH = os.path.join(_CNN_DIR, "cnn_model.onnx")
MEAN_PATH  = os.path.join(_CNN_DIR, "norm_mean.npy")
STD_PATH   = os.path.join(_CNN_DIR, "norm_std.npy")

PREDICTION_LABELS = {
    0: "normal",
    1: "heel_strike",
    2: "overstride",
    3: "foot_drag",
}

_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

_session = None
_mean    = None
_std     = None

try:
    _session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    _mean    = np.load(MEAN_PATH)
    _std     = np.load(STD_PATH)
    print("[predict service] ONNX 모델 로드 성공")
except Exception as e:
    print(f"[predict service] 모델 로드 실패: {e}")


def _butter_lowpass_filter(data, cutoff=20.0, fs=50.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    x = np.array(data, dtype=float)
    return filtfilt(b, a, x)


def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    if _session is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")

    if not request.steps:
        return PredictBatchResponse(status="success", predictions=[])

    signals = []
    for step in request.steps:
        axes = [_butter_lowpass_filter(getattr(step, ax)) for ax in _AXES]
        if any(len(ax) != 100 for ax in axes):
            raise HTTPException(
                status_code=400,
                detail=f"입력 데이터는 100포인트여야 합니다. 현재: {[len(ax) for ax in axes]}",
            )
        signals.append(np.stack(axes, axis=1))  # (100, 6)

    X = np.stack(signals, axis=0).astype(np.float32)  # (N, 100, 6)
    X = (X - _mean) / _std
    X = X.transpose(0, 2, 1)                          # (N, 6, 100)

    preds = _session.run(["output"], {"input": X})[0].argmax(axis=1)

    predictions = [
        StepPrediction(
            prediction_code=int(p),
            prediction_label=PREDICTION_LABELS.get(int(p), "알 수 없음"),
        )
        for p in preds
    ]

    return PredictBatchResponse(status="success", predictions=predictions)
