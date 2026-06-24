import numpy as np
import joblib
import logging
from pathlib import Path
from functools import lru_cache

from fastdtw import fastdtw
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis, skew, entropy as scipy_entropy

from app.schemas.heelstrike import StepRequest, StepResponse

logger = logging.getLogger(__name__)

TARGET_LEN = 100
FILTER_CUTOFF = 40.0
SAMPLE_RATE = 100.0
IMPACT_WINDOW = 12
DTW_RATIO_THRESH = 0.46

class ModelNotLoadedError(RuntimeError):
    pass

_model_package: dict | None = None

def load_model(model_path: str | Path) -> dict:
    global _model_package
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

    logger.info(f"모델 로딩 중: {path}")
    _model_package = joblib.load(path)
    return _model_package

def get_model() -> dict:
    if _model_package is None:
        raise ModelNotLoadedError("모델이 로드되지 않았습니다.")
    return _model_package

# ─────────────────────────────────────────────
# ⚡ 핵심 최적화: 고속 캐싱을 위한 내부 DTW 함수
# ─────────────────────────────────────────────
def _weighted_dist(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * (a - b) ** 2)))

@lru_cache(maxsize=512)  # 👈 최근 512보 동안의 유사한 파형 계산 결과를 메모리에 박아둠
def _cached_fastdtw(xyz_bytes: bytes) -> tuple[float, float, float]:
    """넘파이 바이트 배열을 받아 대기시간이 큰 fastdtw 연산을 캐싱합니다."""
    pkg = get_model()
    # 바이트를 다시 넘파이 배열로 복원
    xyz = np.frombuffer(xyz_bytes, dtype=float).reshape((TARGET_LEN, 4))
    
    dist_fn = lambda a, b: _weighted_dist(a, b, pkg["axis_weights"])
    
    d_h_raw, path_h = fastdtw(xyz, pkg["heel_tmpl"], dist=dist_fn)
    d_n_raw, path_n = fastdtw(xyz, pkg["norm_tmpl"], dist=dist_fn)
    
    d_h = d_h_raw / len(path_h)
    d_n = d_n_raw / len(path_n)
    ratio = d_h / (d_h + d_n + 1e-8)
    return d_h, d_n, ratio

# ─────────────────────────────────────────────
# 전처리 및 피처 추출 (기존과 동일)
# ─────────────────────────────────────────────
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
    if len(zs) < 3: return None
    t_old = np.linspace(0, 1, len(zs))
    t_new = np.linspace(0, 1, TARGET_LEN)
    rx = np.interp(t_new, t_old, xs)
    ry = np.interp(t_new, t_old, ys)
    rz = np.interp(t_new, t_old, zs)
    rmag = np.sqrt(rx**2 + ry**2 + rz**2)
    std_mag = rmag.std() + 1e-8
    return np.column_stack((rx / std_mag, ry / std_mag, rz / std_mag, rmag / std_mag))

def _extract_features(xyz: np.ndarray, contact_time: float) -> dict:
    f: dict = {"contact_time": contact_time}
    axes = ["x", "y", "z", "mag"]
    for i, ax in enumerate(axes):
        sig = xyz[:, i]
        f[f"{ax}_mean"] = np.mean(sig)
        f[f"{ax}_std"] = np.std(sig)
        f[f"{ax}_max"] = np.max(sig)
        f[f"{ax}_min"] = np.min(sig)
        f[f"{ax}_range"] = np.max(sig) - np.min(sig)
        f[f"{ax}_kurtosis"] = kurtosis(sig)
        f[f"{ax}_skewness"] = skew(sig)
        rms = np.sqrt(np.mean(sig**2)) + 1e-8
        f[f"{ax}_crest_factor"] = np.max(np.abs(sig)) / rms
        hist, _ = np.histogram(sig, bins=10, density=True)
        f[f"{ax}_entropy"] = scipy_entropy(hist + 1e-8)
        f[f"{ax}_power"] = np.mean(sig**2)
        fft_v = np.abs(np.fft.rfft(sig))
        f[f"{ax}_fft_mean"] = np.mean(fft_v)
        f[f"{ax}_fft_max"] = np.max(fft_v)
        f[f"{ax}_high_freq_energy"] = np.sum(fft_v[len(fft_v) // 2:])
        jerk = np.diff(sig)
        f[f"{ax}_jerk_mean"] = np.mean(np.abs(jerk))
        f[f"{ax}_jerk_max"] = np.max(np.abs(jerk))
        f[f"{ax}_jerk_std"] = np.std(jerk)
    f["corr_xy"] = np.corrcoef(xyz[:, 0], xyz[:, 1])[0, 1]
    f["corr_xz"] = np.corrcoef(xyz[:, 0], xyz[:, 2])[0, 1]
    f["corr_yz"] = np.corrcoef(xyz[:, 1], xyz[:, 2])[0, 1]
    f["corr_zmag"] = np.corrcoef(xyz[:, 2], xyz[:, 3])[0, 1]
    win = IMPACT_WINDOW
    f["impact_x"] = np.max(np.abs(xyz[:win, 0]))
    f["impact_y"] = np.max(np.abs(xyz[:win, 1]))
    f["impact_z"] = np.max(xyz[:win, 2])
    f["impact_mag"] = np.max(xyz[:win, 3])
    f["landing_slope_z"] = (xyz[5, 2] - xyz[0, 2]) / 5.0
    f["landing_slope_mag"] = (xyz[5, 3] - xyz[0, 3]) / 5.0
    f["impact_ratio_z"] = f["impact_z"] / (np.mean(np.abs(xyz[:, 2])) + 1e-8)
    f["energy_ratio_z"] = np.mean(xyz[TARGET_LEN // 2:, 2] ** 2) / (np.mean(xyz[:TARGET_LEN // 2, 2] ** 2) + 1e-8)
    return f

# ─────────────────────────────────────────────
# 추론 메인 진입부
# ─────────────────────────────────────────────
def predict(request: StepRequest) -> StepResponse:
    pkg = get_model()

    if not request.lengths_match():
        raise ValueError("배열 길이가 다릅니다.")

    proc_x = _preprocess_signal(request.acc_x)
    proc_y = _preprocess_signal(request.acc_y)
    proc_z = _preprocess_signal(request.acc_z)

    xyz = _resample_step(proc_x, proc_y, proc_z, request.contact_time)
    if xyz is None:
        raise ValueError("리샘플링 실패")

    feat_dict = _extract_features(xyz, request.contact_time)

    # 💡 최적화 포인트: 넘파이 배열을 바이트화하여 고속 캐시 레이어 통과
    d_heel, d_norm, ratio = _cached_fastdtw(xyz.tobytes())
    
    feat_dict.update({
        "dtw_dist_heel": d_heel,
        "dtw_dist_normal": d_norm,
        "dtw_ratio": ratio
    })

    full_feat_array = np.array([feat_dict.get(c, 0.0) for c in pkg["feat_cols"]], dtype=float)
    selected = full_feat_array[pkg["top_indices"]].reshape(1, -1)
    scaled = pkg["scaler"].transform(selected)

    svm_prob = float(pkg["svm"].predict_proba(scaled)[0, 1])
    rf_prob = float(pkg["rf"].predict_proba(scaled)[0, 1])
    lgbm_prob = float(pkg["lgbm"].predict_proba(scaled)[0, 1])

    hybrid_prob = lgbm_prob * 0.40 + svm_prob * 0.35 + rf_prob * 0.25
    best_thresh = pkg["best_thresh"]

    is_heel = hybrid_prob >= best_thresh and ratio < DTW_RATIO_THRESH
    label = "heel_strike" if is_heel else "normal"
    message = "힐스트라이크가 감지되었습니다. 발 앞꿈치로 착지하세요." if is_heel else "정상적인 착지 패턴입니다."

    return StepResponse(
        is_heel_strike=is_heel,
        label=label,
        confidence=round(hybrid_prob, 4),
        message=message,
    )