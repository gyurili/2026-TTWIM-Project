import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from app.core.exceptions import ModelNotLoadedError, StepLengthError
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse, StepPrediction
from app.services.gait_features import WINDOW_SIZE, extract_features

logger = logging.getLogger(__name__)

_DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH  = os.path.join(_DATA_DIR, "gait_model.pkl")
SCALER_PATH = os.path.join(_DATA_DIR, "gait_scaler.pkl")
META_PATH   = os.path.join(_DATA_DIR, "gait_meta.json")

PREDICTION_LABELS = {0: "normal", 1: "abnormal"}
_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]   # gait_features 입력 순서

_model         = None
_scaler        = None
_feature_names = None
_meta          = {}


def _warn_version_mismatch(trained: dict) -> None:
    """모델을 학습한 환경과 서버의 라이브러리 버전이 다르면 경고"""
    from importlib.metadata import version, PackageNotFoundError
    for pkg, trained_ver in trained.items():
        try:
            server_ver = version(pkg)
        except PackageNotFoundError:
            server_ver = None
        if trained_ver and server_ver != trained_ver:
            logger.warning("라이브러리 버전 불일치: %s 학습=%s / 서버=%s", pkg, trained_ver, server_ver)


def load_models() -> None:
    global _model, _scaler, _feature_names, _meta
    _model  = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)

    with open(META_PATH, encoding="utf-8") as f:
        _meta = json.load(f)
    _feature_names = list(_meta["features"])

    scaler_names = getattr(_scaler, "feature_names_in_", None)
    if scaler_names is not None and list(scaler_names) != _feature_names:
        raise RuntimeError("gait_meta.json의 features와 scaler의 feature_names_in_이 다릅니다.")

    if _meta.get("window_size", WINDOW_SIZE) != WINDOW_SIZE:
        raise RuntimeError("gait_features.py의 WINDOW_SIZE와 모델 메타의 window_size가 다릅니다.")

    _warn_version_mismatch(_meta.get("library_versions", {}))

    logger.info(
        "모델 로드 성공: %s, feature %d개, 학습일 %s",
        _meta.get("model_name"), len(_feature_names), _meta.get("created_at"),
    )


def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    if _model is None or _scaler is None or _feature_names is None:
        raise ModelNotLoadedError()

    if not request.steps:
        return PredictBatchResponse(predictions=[])

    feat_rows = []
    for i, step in enumerate(request.steps):
        signals = [getattr(step, axis) for axis in _AXES]
        lengths = [len(s) for s in signals]
        if any(l != WINDOW_SIZE for l in lengths):
            raise StepLengthError(
                f"step[{i}] 입력 데이터는 {WINDOW_SIZE}포인트여야 합니다. 현재: {lengths}"
            )
        feat_rows.append(extract_features(*signals))

    X = pd.DataFrame(feat_rows)[_feature_names]
    X_scaled = pd.DataFrame(_scaler.transform(X), columns=_feature_names)
    preds = _model.predict(X_scaled)

    predictions = [
        StepPrediction(
            prediction_code=int(p),
            prediction_label=PREDICTION_LABELS.get(int(p), "알 수 없음"),
        )
        for p in np.asarray(preds).ravel()
    ]

    return PredictBatchResponse(predictions=predictions)