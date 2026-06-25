import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis, skew

from app.schemas.footdrag import (
    FootDragBatchRequest,
    FootDragBatchResponse,
    FootDragPredictionResponse,
    FootDragStepRequest,
)

logger = logging.getLogger(__name__)

TARGET_LEN = 100
FILTER_CUTOFF = 40.0
SAMPLE_RATE = 100.0
IMPACT_WINDOW = 12

_model_package: dict[str, Any] | None = None


class ModelNotLoadedError(RuntimeError):
    pass


def _default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "final_hybrid_model_footdrag_v2.pkl"


def load_model(model_path: str | Path | None = None) -> dict[str, Any]:
    global _model_package

    path = Path(model_path) if model_path is not None else _default_model_path()
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

    logger.info("모델 로딩 중: %s", path)
    _model_package = joblib.load(path)
    return _model_package


def get_model() -> dict[str, Any]:
    if _model_package is None:
        raise ModelNotLoadedError("모델이 로드되지 않았습니다.")
    return _model_package


def _weighted_dist(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * (a - b) ** 2)))


@lru_cache(maxsize=512)
def _cached_fastdtw(xyz_bytes: bytes) -> tuple[float, float, float]:
    pkg = get_model()
    xyz = np.frombuffer(xyz_bytes, dtype=np.float64).reshape((TARGET_LEN, 4))
    axis_weights = np.asarray(pkg["axis_weights"], dtype=float)
    dist_fn = lambda a, b: _weighted_dist(a, b, axis_weights)

    d_drag_raw, path_drag = fastdtw(xyz, pkg["drag_tmpl"], dist=dist_fn)
    d_norm_raw, path_norm = fastdtw(xyz, pkg["norm_tmpl"], dist=dist_fn)

    d_drag = d_drag_raw / max(len(path_drag), 1)
    d_norm = d_norm_raw / max(len(path_norm), 1)
    ratio = d_drag / (d_drag + d_norm + 1e-8)
    return d_drag, d_norm, ratio


def _butter_lowpass(data: np.ndarray, cutoff: float = FILTER_CUTOFF, fs: float = SAMPLE_RATE, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data.astype(float))


def _preprocess_signal(sig: list[float]) -> np.ndarray:
    arr = np.array(sig, dtype=float)
    filtered = _butter_lowpass(arr)
    return filtered - filtered.mean()


def _resample_step(proc_x: np.ndarray, proc_y: np.ndarray, proc_z: np.ndarray, contact_time: float) -> np.ndarray | None:
    ms_per_pt = contact_time / max(len(proc_z), 1)
    half_win = max(3, int(150 / ms_per_pt))
    dyn_h = np.max(proc_z) * 0.5
    peaks, _ = find_peaks(proc_z, height=dyn_h, distance=5)
    center = peaks[np.argmax(proc_z[peaks])] if len(peaks) > 0 else len(proc_z) // 2

    s = max(0, center - half_win)
    e = min(len(proc_z), center + half_win)
    xs, ys, zs = proc_x[s:e], proc_y[s:e], proc_z[s:e]

    if len(zs) < 3:
        return None

    t_old = np.linspace(0, 1, len(zs))
    t_new = np.linspace(0, 1, TARGET_LEN)
    rx = np.interp(t_new, t_old, xs)
    ry = np.interp(t_new, t_old, ys)
    rz = np.interp(t_new, t_old, zs)
    rmag = np.sqrt(rx**2 + ry**2 + rz**2)
    std_mag = rmag.std() + 1e-8
    return np.column_stack((rx / std_mag, ry / std_mag, rz / std_mag, rmag / std_mag))


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    value = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _extract_features(xyz: np.ndarray, contact_time: float) -> dict[str, float]:
    f: dict[str, float] = {"contact_time": float(contact_time)}
    axes = ["x", "y", "z", "mag"]

    for i, ax in enumerate(axes):
        sig = xyz[:, i]
        f[f"{ax}_mean"] = float(np.mean(sig))
        f[f"{ax}_std"] = float(np.std(sig))
        f[f"{ax}_max"] = float(np.max(sig))
        f[f"{ax}_min"] = float(np.min(sig))
        f[f"{ax}_range"] = float(np.max(sig) - np.min(sig))
        f[f"{ax}_kurtosis"] = float(kurtosis(sig))
        f[f"{ax}_skewness"] = float(skew(sig))
        rms = np.sqrt(np.mean(sig**2)) + 1e-8
        f[f"{ax}_crest_factor"] = float(np.max(np.abs(sig)) / rms)
        hist, _ = np.histogram(sig, bins=10, density=True)
        f[f"{ax}_entropy"] = float(scipy_entropy(hist + 1e-8))
        f[f"{ax}_power"] = float(np.mean(sig**2))

        fft_v = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), 1.0 / SAMPLE_RATE)
        f[f"{ax}_fft_mean"] = float(np.mean(fft_v))
        f[f"{ax}_fft_max"] = float(np.max(fft_v))
        f[f"{ax}_high_freq_energy"] = float(np.sum(fft_v[len(fft_v) // 2 :]))

        jerk = np.diff(sig)
        f[f"{ax}_jerk_mean"] = float(np.mean(np.abs(jerk)))
        f[f"{ax}_jerk_max"] = float(np.max(np.abs(jerk)))
        f[f"{ax}_jerk_std"] = float(np.std(jerk))

        autocorr = np.correlate(sig, sig, mode="full")[len(sig) - 1 :]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        for lag in [1, 3, 5]:
            f[f"{ax}_autocorr_{lag}"] = float(autocorr[lag]) if len(autocorr) > lag else 0.0

        zcr = np.sum(np.diff(np.sign(sig)) != 0)
        f[f"{ax}_zcr"] = float(zcr / max(len(sig), 1))

        f[f"{ax}_dominant_freq"] = float(freqs[np.argmax(fft_v)])
        f[f"{ax}_spectral_centroid"] = float(np.sum(freqs * fft_v) / (np.sum(fft_v) + 1e-8))

        n = len(sig)
        t = n // 3
        e_mean = float(np.mean(np.abs(sig[:t])))
        m_mean = float(np.mean(np.abs(sig[t : 2 * t])))
        l_mean = float(np.mean(np.abs(sig[2 * t :])))
        f[f"{ax}_early_mean"] = e_mean
        f[f"{ax}_mid_mean"] = m_mean
        f[f"{ax}_late_mean"] = l_mean
        f[f"{ax}_early_late_ratio"] = float(e_mean / (l_mean + 1e-8))
        f[f"{ax}_mid_late_ratio"] = float(m_mean / (l_mean + 1e-8))

    x_sig, y_sig, z_sig = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    f["corr_xy"] = _safe_corrcoef(xyz[:, 0], xyz[:, 1])
    f["corr_xz"] = _safe_corrcoef(xyz[:, 0], xyz[:, 2])
    f["corr_yz"] = _safe_corrcoef(xyz[:, 1], xyz[:, 2])
    f["corr_zmag"] = _safe_corrcoef(xyz[:, 2], xyz[:, 3])

    win = min(IMPACT_WINDOW, len(xyz))
    f["impact_x"] = float(np.max(np.abs(xyz[:win, 0])))
    f["impact_y"] = float(np.max(np.abs(xyz[:win, 1])))
    f["impact_z"] = float(np.max(xyz[:win, 2]))
    f["impact_mag"] = float(np.max(xyz[:win, 3]))

    z_index = min(5, len(xyz) - 1)
    slope_divisor = max(z_index, 1)
    f["landing_slope_z"] = float((xyz[z_index, 2] - xyz[0, 2]) / slope_divisor)
    f["landing_slope_mag"] = float((xyz[z_index, 3] - xyz[0, 3]) / slope_divisor)
    f["impact_ratio_z"] = float(f["impact_z"] / (np.mean(np.abs(xyz[:, 2])) + 1e-8))
    f["energy_ratio_z"] = float(
        np.mean(xyz[TARGET_LEN // 2 :, 2] ** 2) / (np.mean(xyz[: TARGET_LEN // 2, 2] ** 2) + 1e-8)
    )

    fft_x = np.abs(np.fft.rfft(x_sig))
    fft_y = np.abs(np.fft.rfft(y_sig))
    f["horiz_high_freq_ratio"] = float(
        (np.sum(fft_x[len(fft_x) // 2 :]) + np.sum(fft_y[len(fft_y) // 2 :]))
        / (np.sum(fft_x) + np.sum(fft_y) + 1e-8)
    )
    f["mid_late_jerk_mean_x"] = float(np.mean(np.abs(np.diff(x_sig[win:]))) if len(x_sig[win:]) > 1 else 0.0)
    f["mid_late_jerk_mean_y"] = float(np.mean(np.abs(np.diff(y_sig[win:]))) if len(y_sig[win:]) > 1 else 0.0)
    f["z_max_to_horiz_range"] = float(
        np.max(z_sig) / (np.max(np.abs(x_sig)) + np.max(np.abs(y_sig)) + 1e-8)
    )

    return f


def _predict_one(request: FootDragStepRequest) -> FootDragPredictionResponse:
    pkg = get_model()

    if not request.lengths_match():
        raise ValueError("acc_x, acc_y, acc_z 길이가 일치하지 않습니다.")

    proc_x = _preprocess_signal(request.acc_x)
    proc_y = _preprocess_signal(request.acc_y)
    proc_z = _preprocess_signal(request.acc_z)

    xyz = _resample_step(proc_x, proc_y, proc_z, request.contact_time)
    if xyz is None:
        raise ValueError("리샘플링 실패")

    feat_dict = _extract_features(xyz, request.contact_time)
    d_drag, d_norm, ratio = _cached_fastdtw(xyz.tobytes())
    feat_dict.update({"dist_drag": d_drag, "dist_normal": d_norm, "ratio": ratio})

    feature_names = list(pkg["feat_cols"]) + list(pkg["dtw_feats"])
    full_feat_array = np.array([feat_dict.get(name, 0.0) for name in feature_names], dtype=float)
    top_indices = np.asarray(pkg["top_indices"], dtype=int)
    selected_feature_names = [feature_names[index] for index in top_indices]
    selected_array = full_feat_array[top_indices].reshape(1, -1)
    scaled_array = pkg["scaler"].transform(selected_array)
    scaled_frame = pd.DataFrame(scaled_array, columns=selected_feature_names)

    lgbm_prob = float(pkg["lgbm"].predict_proba(scaled_frame)[0, 1])
    xgb_prob = float(pkg["xgb"].predict_proba(scaled_array)[0, 1])
    svm_prob = float(pkg["svm"].predict_proba(scaled_array)[0, 1])
    rf_prob = float(pkg["rf"].predict_proba(scaled_array)[0, 1])

    hybrid = np.array([[lgbm_prob, xgb_prob, svm_prob, rf_prob]], dtype=float)
    confidence = float(pkg["meta"].predict_proba(hybrid)[0, 1])
    model_threshold = float(pkg["best_thresh"])
    dtw_threshold = float(pkg.get("dtw_thresh", 0.5))
    is_foot_drag = confidence >= model_threshold and ratio < dtw_threshold

    prediction_label = "발끌기" if is_foot_drag else "정상"
    message = "발끌기가 감지되었습니다. 착지 패턴을 점검하세요." if is_foot_drag else "정상적인 러닝 패턴입니다."

    return FootDragPredictionResponse(
        prediction_code=int(is_foot_drag),
        prediction_label=prediction_label,
        confidence=round(confidence, 4),
        is_foot_drag=is_foot_drag,
        message=message,
        dtw_ratio=round(float(ratio), 4),
        dtw_threshold=dtw_threshold,
        model_threshold=model_threshold,
    )


def predict_step(request: FootDragStepRequest) -> FootDragPredictionResponse:
    return _predict_one(request)


def predict_batch(request: FootDragBatchRequest) -> FootDragBatchResponse:
    predictions = [_predict_one(step) for step in request.steps]
    return FootDragBatchResponse(status="success", predictions=predictions)
