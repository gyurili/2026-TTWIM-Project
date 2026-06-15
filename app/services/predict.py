import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from app.schemas.predict import PredictRequest, PredictResponse

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ensemble_model.pkl")

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


def _extract_features(x_arr, y_arr, z_arr, contact_time):
    filt_x = _butter_lowpass_filter(x_arr)
    filt_y = _butter_lowpass_filter(y_arr)
    filt_z = _butter_lowpass_filter(z_arr)

    max_z = np.max(filt_z)
    time_to_peak_z = np.argmax(filt_z) / len(filt_z)
    max_jerk_z = np.max(np.gradient(filt_z))
    std_z = np.std(filt_z)

    max_abs_y = np.max(np.abs(filt_y))
    std_y = np.std(filt_y)

    max_abs_x = np.max(np.abs(filt_x))
    std_x = np.std(filt_x)

    vector_magnitude = np.sqrt(filt_x**2 + filt_y**2 + filt_z**2)
    max_vm = np.max(vector_magnitude)

    ratio_z_y = max_z / (max_abs_y + 1e-6)
    energy_z = np.sum(filt_z**2)

    return pd.DataFrame([{
        "max_z": max_z,
        "time_to_peak_z": time_to_peak_z,
        "max_jerk_z": max_jerk_z,
        "std_z": std_z,
        "max_abs_y": max_abs_y,
        "std_y": std_y,
        "max_abs_x": max_abs_x,
        "std_x": std_x,
        "max_vm": max_vm,
        "ratio_z_y": ratio_z_y,
        "energy_z": energy_z,
        "contact_time": contact_time,
    }])


def predict(data: PredictRequest) -> PredictResponse:
    if _model is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")

    if len(data.acc_z) < 10:
        raise ValueError("데이터 길이가 너무 짧습니다.")

    features_df = _extract_features(data.acc_x, data.acc_y, data.acc_z, data.contact_time)

    prediction = _model.predict(features_df)[0]
    label = "정상 보행" if prediction == 0 else "힐스트라이크"

    return PredictResponse(
        status="success",
        prediction_code=int(prediction),
        prediction_label=label,
    )
