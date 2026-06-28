from fastapi import APIRouter, HTTPException
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse
from app.services import predict as service

router = APIRouter(tags=["Predict"])


@router.post("/predict", response_model=PredictBatchResponse)
def predict_heel_strike(data: PredictBatchRequest):
    try:
        return service.predict_batch(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ML Integration: running normal/abnormal classification endpoint
# Existing endpoints above remain unchanged.
# =============================================================================
import os
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, Field
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew


ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "running_pipeline.pkl")


class MLStepRequest(BaseModel):
    raw_acc_x: list[float] = Field(..., min_length=200, max_length=200)
    raw_acc_y: list[float] = Field(..., min_length=200, max_length=200)
    raw_acc_z: list[float] = Field(..., min_length=200, max_length=200)
    raw_gyro_x: list[float] = Field(..., min_length=200, max_length=200)
    raw_gyro_y: list[float] = Field(..., min_length=200, max_length=200)
    raw_gyro_z: list[float] = Field(..., min_length=200, max_length=200)


class MLPredictRequest(BaseModel):
    records: list[MLStepRequest] = Field(..., min_length=1)


class MLPredictionItem(BaseModel):
    index: int
    prediction_code: int
    prediction_label: str


class MLPredictResponse(BaseModel):
    status: str
    result_code: int
    result_label: str
    predictions: list[MLPredictionItem]


_ml_model = None
try:
    _ml_model = joblib.load(ML_MODEL_PATH)
    print("[predict_ml] running_pipeline.pkl 모델 로드 성공")
except Exception as exc:
    print(f"[predict_ml] 모델 로드 실패: {exc}")


def _ml_lowpass_filter(values: list[float], cutoff: float = 20.0, fs: float = 50.0, order: int = 4) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.size < 2:
        return data
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    padlen = min(len(data) - 1, max(len(a), len(b)) * 3)
    if padlen <= 0:
        return data
    return filtfilt(b, a, data, padlen=padlen)


def _ml_extract_features(record: MLStepRequest) -> np.ndarray:
    axes = [
        _ml_lowpass_filter(record.raw_acc_x),
        _ml_lowpass_filter(record.raw_acc_y),
        _ml_lowpass_filter(record.raw_acc_z),
        _ml_lowpass_filter(record.raw_gyro_x),
        _ml_lowpass_filter(record.raw_gyro_y),
        _ml_lowpass_filter(record.raw_gyro_z),
    ]

    acc_mag = np.sqrt(axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2)
    gyro_mag = np.sqrt(axes[3] ** 2 + axes[4] ** 2 + axes[5] ** 2)

    feats: list[float] = []

    for signal in axes:
        feats.extend(
            [
                float(np.mean(signal)),
                float(np.std(signal)),
                float(np.abs(signal).mean()),
                float(np.min(signal)),
                float(np.max(signal)),
                float(np.max(signal) - np.min(signal)),
                float(skew(signal)),
                float(kurtosis(signal)),
                float(np.sqrt(np.mean(signal**2))),
                float(np.percentile(signal, 25)),
                float(np.percentile(signal, 75)),
                float(np.sum(np.diff(np.sign(signal)) != 0) / 200.0),
            ]
        )

        freqs, psd = welch(signal, fs=100, nperseg=50)
        total_power = float(psd.sum() + 1e-8)
        feats.extend(
            [
                total_power,
                float(freqs[np.argmax(psd)]),
                float(np.sum(freqs * psd) / total_power),
            ]
        )

        t = 200 // 3
        early_mean = float(np.mean(np.abs(signal[:t])))
        late_mean = float(np.mean(np.abs(signal[2 * t:]))) + 1e-8
        feats.extend([early_mean, late_mean, float(early_mean / late_mean)])

        autocorr = np.correlate(signal, signal, mode="full")[len(signal) - 1 :]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        feats.extend([float(autocorr[3]), float(autocorr[5])])

    for left in range(6):
        for right in range(left + 1, 6):
            corr = np.corrcoef(axes[left], axes[right])[0, 1]
            feats.append(float(np.nan_to_num(corr)))

    feats.extend(
        [
            float(np.max(acc_mag[:15])),
            float(np.mean(acc_mag)),
        ]
    )

    return np.asarray(feats, dtype=np.float32)


def _ml_predict_label(prediction_code: int) -> str:
    return "정상" if prediction_code == 0 else "비정상"


@router.post("/predict_ml", response_model=MLPredictResponse, status_code=200)
def predict_ml(data: MLPredictRequest):
    if _ml_model is None:
        raise HTTPException(status_code=503, detail="ML 모델이 로드되지 않았습니다.")

    feature_rows = []
    for idx, record in enumerate(data.records):
        feature_rows.append(_ml_extract_features(record))

    X = np.asarray(feature_rows, dtype=np.float32)
    raw_preds: Any = _ml_model.predict(X)

    predictions = [
        MLPredictionItem(
            index=index,
            prediction_code=int(pred),
            prediction_label=_ml_predict_label(int(pred)),
        )
        for index, pred in enumerate(raw_preds)
    ]

    abnormal_count = sum(item.prediction_code == 1 for item in predictions)
    overall_code = 1 if abnormal_count > 0 else 0

    return MLPredictResponse(
        status="success",
        result_code=overall_code,
        result_label=_ml_predict_label(overall_code),
        predictions=predictions,
    )
