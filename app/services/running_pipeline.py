import os
import ast
import numpy as np
import joblib
from fastapi import HTTPException
from scipy import stats
from scipy.signal import welch
from app.schemas.running_pipeline import (
    RunningPipelineBatchRequest,
    RunningPipelineBatchResponse,
    StepPrediction,
)

_PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "running_pipeline.pkl")

PREDICTION_LABELS = {0: "normal", 1: "abnormal"}
_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

_pipeline = None

try:
    _pipeline = joblib.load(_PIPELINE_PATH)
    print(f"[running_pipeline service] 모델 로드 성공 (threshold={_pipeline.threshold_:.3f})")
except Exception as e:
    print(f"[running_pipeline service] 모델 로드 실패: {e}")


def _extract_features(raw: np.ndarray) -> np.ndarray:
    N = raw.shape[0]
    feat_list = []

    for i in range(N):
        row_feats = []
        for j in range(6):
            x = raw[i, :, j]

            row_feats += [
                x.mean(),
                x.std(),
                np.abs(x).mean(),
                x.min(),
                x.max(),
                x.max() - x.min(),
                float(stats.skew(x)),
                float(stats.kurtosis(x)),
                np.sqrt(np.mean(x**2)),
                float(np.percentile(x, 25)),
                float(np.percentile(x, 75)),
                np.sum(np.diff(np.sign(x)) != 0) / 200.0,
            ]

            freqs, psd = welch(x, fs=100, nperseg=50)
            total_power = psd.sum() + 1e-8
            row_feats += [
                total_power,
                freqs[np.argmax(psd)],
                np.sum(freqs * psd) / total_power,
            ]

            t = 200 // 3
            e_mean = np.mean(np.abs(x[:t]))
            l_mean = np.mean(np.abs(x[2 * t :])) + 1e-8
            row_feats += [e_mean, l_mean, e_mean / l_mean]

            autocorr = np.correlate(x, x, mode="full")[len(x) - 1 :]
            autocorr = autocorr / (autocorr[0] + 1e-8)
            row_feats += [autocorr[3], autocorr[5]]

        for c1 in range(6):
            for c2 in range(c1 + 1, 6):
                row_feats.append(np.corrcoef(raw[i, :, c1], raw[i, :, c2])[0, 1])

        win = 15
        acc_mag = np.sqrt((raw[i, :, :3] ** 2).sum(axis=1))
        row_feats += [
            np.max(acc_mag[:win]),
            np.mean(acc_mag),
        ]

        feat_list.append(row_feats)

    return np.array(feat_list, dtype=np.float32)


def predict_batch(request: RunningPipelineBatchRequest) -> RunningPipelineBatchResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    if not request.steps:
        return RunningPipelineBatchResponse(status="success", predictions=[])

    raw_signals = []
    for i, step in enumerate(request.steps):
        lengths = [len(getattr(step, ax)) for ax in _AXES]
        if any(l != 200 for l in lengths):
            raise HTTPException(
                status_code=400,
                detail=f"step[{i}] 입력 데이터는 200포인트여야 합니다. 현재: {lengths}",
            )
        signal = np.column_stack([
            np.array(getattr(step, ax), dtype=float) for ax in _AXES
        ])
        raw_signals.append(signal)

    X_raw = np.array(raw_signals, dtype=np.float32)
    X_feat = _extract_features(X_raw)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    probs = _pipeline.predict_proba(X_feat)[:, 1]
    preds = (probs >= _pipeline.threshold_).astype(int)

    predictions = [
        StepPrediction(
            prediction_code=int(p),
            prediction_label=PREDICTION_LABELS.get(int(p), "알 수 없음"),
        )
        for p in preds
    ]

    return RunningPipelineBatchResponse(status="success", predictions=predictions)
