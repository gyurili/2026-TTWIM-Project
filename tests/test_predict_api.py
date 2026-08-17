import numpy as np
import pytest

from app.api.predict import _latest_abnormal_run_indices
from app.schemas.predict import StepPrediction
from app.services import coaching as coaching_service
from tests.conftest import signup_and_login

_N = 200
_SR = 50.0


def _build_impact_step(seed):
    """실제 학습된 SVM 모델이 'abnormal'로 판정하도록 튜닝된 착지 충격 패턴.
    (Postman으로 수동 검증했던 것과 동일한 생성 로직)"""
    r = np.random.default_rng(seed)
    t = np.arange(_N) / _SR
    step_hz, gyro_amp, spike_height, spike_width = 2.8, 120, 3.0, 2

    acc_z = 1.0 + 0.15 * np.sin(2 * np.pi * step_hz * t) + r.normal(0, 0.045, _N)
    for p in np.arange(0, _N, _SR / step_hz).astype(int):
        for w in range(spike_width):
            if p + w < _N:
                acc_z[p + w] += spike_height * (1 - w / spike_width)

    return {
        "acc_x": r.normal(0, 0.15, _N).tolist(),
        "acc_y": r.normal(0, 0.15, _N).tolist(),
        "acc_z": acc_z.tolist(),
        "gyro_x": (gyro_amp * np.sin(2 * np.pi * step_hz * t) + r.normal(0, gyro_amp * 0.15, _N)).tolist(),
        "gyro_y": (gyro_amp * np.cos(2 * np.pi * step_hz * t) + r.normal(0, gyro_amp * 0.15, _N)).tolist(),
        "gyro_z": (0.5 * gyro_amp * np.sin(2 * np.pi * step_hz * t + 1) + r.normal(0, gyro_amp * 0.1, _N)).tolist(),
    }


def _build_normal_step(seed):
    r = np.random.default_rng(seed)
    return {
        "acc_x": r.normal(0, 0.05, _N).tolist(),
        "acc_y": r.normal(0, 0.05, _N).tolist(),
        "acc_z": (1.0 + r.normal(0, 0.02, _N)).tolist(),
        "gyro_x": r.normal(0, 2, _N).tolist(),
        "gyro_y": r.normal(0, 2, _N).tolist(),
        "gyro_z": r.normal(0, 2, _N).tolist(),
    }


def _predict_payload(steps):
    return {
        "steps": steps,
        "dtwContext": {
            "status": "alert",
            "avgSimilarity": 0.55,
            "trend": "worsening",
            "shapeDev": 1.2,
            "cadenceDev": 0.8,
            "amplitudeDev": 3.4,
            "timeline": [0.7, 0.62, 0.58, 0.5],
        },
        "altitudeContext": {"readings": [0.1, 0.15, 0.1]},
    }


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        ([1, 0, 1, 0, 1], [4]),
        ([0, 0, 1, 1, 1], [2, 3, 4]),
        ([1, 0, 1, 1, 0], [2, 3]),
        ([1, 1, 1], [0, 1, 2]),
        ([0, 0, 0], []),
    ],
)
def test_latest_abnormal_run_indices_uses_only_most_recent_contiguous_run(codes, expected):
    predictions = [
        StepPrediction(prediction_code=code, prediction_label="abnormal" if code else "normal")
        for code in codes
    ]

    assert _latest_abnormal_run_indices(predictions) == expected


def test_predict_requires_token(client):
    res = client.post("/predict", json=_predict_payload([_build_normal_step(1)]))

    assert res.status_code == 401


def test_predict_wrong_step_length_rejected(client):
    _, headers = signup_and_login(client)
    bad_step = _build_normal_step(1)
    bad_step["acc_x"] = bad_step["acc_x"][:100]  # 200포인트여야 하는데 100개만 보냄

    res = client.post("/predict", json=_predict_payload([bad_step]), headers=headers)

    assert res.status_code == 400


def test_predict_mostly_normal_steps_has_no_coaching(client):
    """비정상 판정이 과반 미만이면 코칭 텍스트를 생성하지 않아야 한다 (LLM 호출 자체가 없어야 함)."""
    _, headers = signup_and_login(client)
    steps = [_build_normal_step(s) for s in (1, 2, 3)]

    res = client.post("/predict", json=_predict_payload(steps), headers=headers)

    assert res.status_code == 200
    data = res.json()["data"]
    assert len(data["predictions"]) == 3
    assert data["coaching_text"] == ""
    assert data["coaching_tags"] == []


def test_predict_mostly_abnormal_steps_triggers_mocked_coaching(client, monkeypatch):
    """비정상 판정이 과반이면 코칭이 트리거되는데, 실제 LLM은 mock으로 대체해서 검증한다."""
    _, headers = signup_and_login(client)
    steps = [_build_impact_step(s) for s in (11, 21, 31)]

    call_count = {"n": 0}

    def _fake_call_llm(prompt, system_instruction):
        call_count["n"] += 1
        return "무릎을 부드럽게 굽혀 충격을 흡수해 보세요.", ["충격 과다 러닝"]

    monkeypatch.setattr(coaching_service, "_call_llm", _fake_call_llm)

    res = client.post("/predict", json=_predict_payload(steps), headers=headers)

    assert res.status_code == 200
    data = res.json()["data"]
    assert sum(p["prediction_code"] for p in data["predictions"]) >= 2  # 3개 중 과반 abnormal
    assert data["coaching_text"] == "무릎을 부드럽게 굽혀 충격을 흡수해 보세요."
    assert data["coaching_tags"] == ["충격 과다 러닝"]
    assert call_count["n"] == 1
