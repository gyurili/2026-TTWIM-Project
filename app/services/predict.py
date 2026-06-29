import logging
import os

import numpy as np
import joblib
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
from fastapi import HTTPException
from app.core.exceptions import ModelNotLoadedError
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse, StepPrediction

logger = logging.getLogger(__name__)

_DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH  = os.path.join(_DATA_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(_DATA_DIR, "svm_scaler.pkl")

PREDICTION_LABELS = {0: "normal", 1: "abnormal"}
_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

_model         = None
_scaler        = None
_feature_names = None


def load_models() -> None:
    global _model, _scaler, _feature_names
    _model  = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    if hasattr(_scaler, "feature_names_in_"):
        _feature_names = _scaler.feature_names_in_
    else:
        logger.warning("스케일러에 feature_names_in_ 속성이 없습니다.")
    logger.info("SVM 모델 및 스케일러 로드 성공")


def _extract_features(step) -> dict:
    b, a = butter(4, 20 / 25, btype="low")

    signals = {}
    for axis in _AXES:
        raw = np.array(getattr(step, axis), dtype=float)
        signals[axis] = filtfilt(b, a, raw)

    ax_arr, ay_arr, az_arr = signals["acc_x"], signals["acc_y"], signals["acc_z"]
    gx_arr, gy_arr, gz_arr = signals["gyro_x"], signals["gyro_y"], signals["gyro_z"]

    acc_mag  = np.sqrt(ax_arr**2 + ay_arr**2 + az_arr**2)
    gyro_mag = np.sqrt(gx_arr**2 + gy_arr**2 + gz_arr**2)

    feats = {}

    for name, sig in [("ax", ax_arr), ("ay", ay_arr), ("az", az_arr),
                    ("gx", gx_arr), ("gy", gy_arr), ("gz", gz_arr)]:
        feats[f"{name}_mean"] = np.mean(sig)
        feats[f"{name}_std"]  = np.std(sig)
        feats[f"{name}_max"]  = np.max(sig)
        feats[f"{name}_min"]  = np.min(sig)
        feats[f"{name}_skew"] = float(skew(sig))
        feats[f"{name}_kurt"] = float(kurtosis(sig))

    for name, sig in [("acc_mag", acc_mag), ("gyro_mag", gyro_mag)]:
        feats[f"{name}_mean"] = np.mean(sig)
        feats[f"{name}_std"]  = np.std(sig)
        feats[f"{name}_max"]  = np.max(sig)
        feats[f"{name}_min"]  = np.min(sig)

    peaks, props = find_peaks(az_arr, height=0.5, distance=15)
    feats["az_peak_count"]    = len(peaks)
    feats["az_peak_mean_val"] = np.mean(props["peak_heights"]) if len(peaks) > 0 else 0
    feats["az_peak_std_val"]  = np.std(props["peak_heights"])  if len(peaks) > 1 else 0
    if len(peaks) > 1:
        intervals = np.diff(peaks)
        feats["az_peak_interval_mean"] = np.mean(intervals)
        feats["az_peak_interval_std"]  = np.std(intervals)
        feats["az_peak_interval_cv"]   = np.std(intervals) / (np.mean(intervals) + 1e-6)
    else:
        feats["az_peak_interval_mean"] = 0
        feats["az_peak_interval_std"]  = 0
        feats["az_peak_interval_cv"]   = 0

    feats["az_rms"]       = np.sqrt(np.mean(az_arr**2))
    feats["acc_mag_rms"]  = np.sqrt(np.mean(acc_mag**2))
    feats["gyro_mag_rms"] = np.sqrt(np.mean(gyro_mag**2))

    feats["az_front_back_ratio"]       = np.mean(az_arr[:100]**2) / (np.mean(az_arr[100:]**2) + 1e-6)
    feats["gyro_mag_front_back_ratio"] = np.mean(gyro_mag[:100]**2) / (np.mean(gyro_mag[100:]**2) + 1e-6)

    feats["ax_az_corr"] = float(np.corrcoef(ax_arr, az_arr)[0, 1])
    feats["gx_gz_corr"] = float(np.corrcoef(gx_arr, gz_arr)[0, 1])

    return feats


def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    if _model is None or _scaler is None:
        raise ModelNotLoadedError()

    if not request.steps:
        return PredictBatchResponse(status="success", predictions=[])

    feat_rows = []
    for i, step in enumerate(request.steps):
        lengths = [len(getattr(step, axis)) for axis in _AXES]
        if any(l != 200 for l in lengths):
            raise HTTPException(
                status_code=400,
                detail=f"step[{i}] 입력 데이터는 200포인트여야 합니다. 현재: {lengths}",
            )
        feat_rows.append(_extract_features(step))

    X = pd.DataFrame(feat_rows)
    if _feature_names is not None:
        X = X[_feature_names]

    X_scaled = pd.DataFrame(_scaler.transform(X), columns=X.columns)
    preds = _model.predict(X_scaled)

    predictions = [
        StepPrediction(
            prediction_code=int(p),
            prediction_label=PREDICTION_LABELS.get(int(p), "알 수 없음"),
        )
        for p in preds
    ]

    return PredictBatchResponse(status="success", predictions=predictions)
