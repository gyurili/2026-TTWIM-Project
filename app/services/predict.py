import os
import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException
from scipy.signal import butter, filtfilt
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse, StepPrediction

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cnn_model.pt")

PREDICTION_LABELS = {
    0: "normal",
    1: "heel_strike",
    2: "overstride",
    3: "foot_drag",
}

_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


class GaitCNN(nn.Module):
    def __init__(self, n_channels=6, n_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv(x))


def _butter_lowpass_filter(data, cutoff=40.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    x = np.array(data, dtype=float)
    return filtfilt(b, a, x)


_model = None
_mean = None
_std = None
_device = torch.device("cpu")

try:
    checkpoint = torch.load(MODEL_PATH, map_location=_device)
    _model = GaitCNN().to(_device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    _mean = checkpoint["mean"]
    _std = checkpoint["std"]
    print("[predict service] CNN 모델 로드 성공")
except Exception as e:
    print(f"[predict service] 모델 로드 실패: {e}")


def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    if _model is None:
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
        signals.append(np.stack(axes, axis=1))

    X = np.stack(signals, axis=0).astype(np.float32)  # (N, 100, 6)
    X = (X - _mean) / _std
    X = X.transpose(0, 2, 1)  # (N, 6, 100)

    tensor = torch.tensor(X, dtype=torch.float32).to(_device)
    with torch.no_grad():
        preds = _model(tensor).argmax(dim=1).cpu().numpy()

    predictions = [
        StepPrediction(
            prediction_code=int(p),
            prediction_label=PREDICTION_LABELS.get(int(p), "알 수 없음"),
        )
        for p in preds
    ]

    return PredictBatchResponse(status="success", predictions=predictions)
