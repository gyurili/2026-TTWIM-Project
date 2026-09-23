import math

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis

WINDOW_SIZE = 200     # 윈도우 길이 (포인트)
FS = 50               # 샘플링 주파수 (Hz)
CUTOFF_HZ = 20        # Low-pass cutoff (Hz)

# 입력 순서: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
_B, _A = butter(4, CUTOFF_HZ / (FS / 2), btype='low')


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """상수 신호일 때 np.corrcoef가 NaN을 내는 것을 방지."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) -> dict:
    raw = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    sig = []
    for r in raw:
        arr = np.asarray(r, dtype=np.float64)
        if arr.shape != (WINDOW_SIZE,):
            raise ValueError(f'윈도우 길이는 {WINDOW_SIZE}이어야 합니다. 현재: {arr.shape}')
        sig.append(filtfilt(_B, _A, arr))

    ax, ay, az, gx, gy, gz = sig
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    feats = {}

    # 1) 축별 기본 통계 (6축 x 6 = 36)
    for name, s in [('ax', ax), ('ay', ay), ('az', az),
                    ('gx', gx), ('gy', gy), ('gz', gz)]:
        feats[f'{name}_mean'] = np.mean(s)
        feats[f'{name}_std'] = np.std(s)
        feats[f'{name}_max'] = np.max(s)
        feats[f'{name}_min'] = np.min(s)
        feats[f'{name}_skew'] = skew(s)
        feats[f'{name}_kurt'] = kurtosis(s)

    # 2) 합성 크기 통계 (2 x 4 = 8)
    for name, s in [('acc_mag', acc_mag), ('gyro_mag', gyro_mag)]:
        feats[f'{name}_mean'] = np.mean(s)
        feats[f'{name}_std'] = np.std(s)
        feats[f'{name}_max'] = np.max(s)
        feats[f'{name}_min'] = np.min(s)

    # 3) Z축 Peak 특성 (6)
    peaks, props = find_peaks(az, height=0.5, distance=15)
    feats['az_peak_count'] = len(peaks)
    feats['az_peak_mean_val'] = np.mean(props['peak_heights']) if len(peaks) > 0 else 0
    feats['az_peak_std_val'] = np.std(props['peak_heights']) if len(peaks) > 1 else 0
    if len(peaks) > 1:
        intervals = np.diff(peaks)
        feats['az_peak_interval_mean'] = np.mean(intervals)
        feats['az_peak_interval_std'] = np.std(intervals)
        feats['az_peak_interval_cv'] = np.std(intervals) / (np.mean(intervals) + 1e-6)
    else:
        feats['az_peak_interval_mean'] = 0
        feats['az_peak_interval_std'] = 0
        feats['az_peak_interval_cv'] = 0

    # 4) RMS (3)
    feats['az_rms'] = np.sqrt(np.mean(az**2))
    feats['acc_mag_rms'] = np.sqrt(np.mean(acc_mag**2))
    feats['gyro_mag_rms'] = np.sqrt(np.mean(gyro_mag**2))

    # 5) 전/후반 에너지 비율 (2)
    half = WINDOW_SIZE // 2
    feats['az_front_back_ratio'] = np.mean(az[:half]**2) / (np.mean(az[half:]**2) + 1e-6)
    feats['gyro_mag_front_back_ratio'] = (np.mean(gyro_mag[:half]**2)
                                          / (np.mean(gyro_mag[half:]**2) + 1e-6))

    # 6) 축 간 상관 (2)
    feats['ax_az_corr'] = _safe_corr(ax, az)
    feats['gx_gz_corr'] = _safe_corr(gx, gz)

    # NaN/inf(상수 신호의 skew/kurt 등) → 0 으로 통일
    return {k: (float(v) if math.isfinite(v) else 0.0) for k, v in feats.items()}


# Feature 이름 목록 (순서 고정, 57개)
FEATURE_NAMES = list(extract_features(
    *np.random.default_rng(0).normal(size=(6, WINDOW_SIZE))).keys())
assert len(FEATURE_NAMES) == 57