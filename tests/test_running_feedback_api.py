from app.services import feedback as feedback_service
from tests.conftest import signup_and_login

_STARTED_AT = 1_700_000_000_000
_ENDED_AT = _STARTED_AT + 5 * 60 * 1000

_MOCK_FEEDBACK = feedback_service._FeedbackOutput(
    summary="초반엔 안정적이었지만 후반부에 리듬이 흐트러졌습니다.",
    strengths=["초반 페이스를 일정하게 유지했어요."],
    improvements=["후반부에 착지 충격이 다소 커졌어요."],
    next_step="다음 러닝에서는 후반 페이스 조절에 집중해보세요.",
)


def _create_session(
    client,
    headers,
    tags=("안정 러닝",),
    window_start_time=None,
    window_end_time=None,
):
    window_start_time = _STARTED_AT if window_start_time is None else window_start_time
    window_end_time = _STARTED_AT + 10_000 if window_end_time is None else window_end_time
    payload = {
        "mode": "outdoor",
        "route": None,
        "startedAt": _STARTED_AT,
        "endedAt": _ENDED_AT,
        "durationSec": 300,
        "distanceMeters": 800.0,
        "windows": [
            {
                "predictions": ["normal", "normal"],
                "tags": list(tags),
                "feedbackText": None,
                "dtwContext": None,
                "windowStartTime": window_start_time,
                "windowEndTime": window_end_time,
            }
        ],
    }
    res = client.post("/running/complete", json=payload, headers=headers)
    assert res.status_code == 200
    return res.json()["data"]["sessionId"]


def _create_session_with_windows(client, headers, window_tags, interval_ms=20_000, window_len_ms=10_000):
    """window_tags: window 하나당 태그 리스트를 순서대로 담은 리스트. _TIMELINE_BATCH_SIZE(5) 단위
    다수결 병합을 테스트하기 위해 window를 여러 개 실어 보내는 헬퍼."""
    windows = []
    t = _STARTED_AT
    for tags in window_tags:
        windows.append({
            "predictions": ["normal"],
            "tags": list(tags),
            "feedbackText": None,
            "dtwContext": None,
            "windowStartTime": t,
            "windowEndTime": t + window_len_ms,
        })
        t += interval_ms
    payload = {
        "mode": "outdoor",
        "route": None,
        "startedAt": _STARTED_AT,
        "endedAt": _ENDED_AT,
        "durationSec": (_ENDED_AT - _STARTED_AT) // 1000,
        "distanceMeters": 800.0,
        "windows": windows,
    }
    res = client.post("/running/complete", json=payload, headers=headers)
    assert res.status_code == 200
    return res.json()["data"]["sessionId"]


def test_generate_feedback_with_mocked_llm(client, monkeypatch):
    """실제 LLM을 호출하지 않고, 응답을 mock으로 대체해서 피드백 생성 흐름만 검증한다."""
    _, headers = signup_and_login(client)
    session_id = _create_session(client, headers)

    call_count = {"n": 0}

    def _fake_call_llm(prompt):
        call_count["n"] += 1
        return _MOCK_FEEDBACK

    monkeypatch.setattr(feedback_service, "_call_llm", _fake_call_llm)

    res = client.post(f"/running/{session_id}/feedback", headers=headers)

    assert res.status_code == 200
    data = res.json()["data"]
    assert data["finalFeedback"]["summary"] == _MOCK_FEEDBACK.summary
    assert data["finalFeedback"]["nextStep"] == _MOCK_FEEDBACK.next_step
    assert data["tagRatio"] == {"안정 러닝": 1.0}
    assert data["sessionSummary"]["distanceKm"] == 0.8
    assert call_count["n"] == 1


def test_feedback_timeline_uses_earliest_window_when_window_precedes_session(client, monkeypatch):
    """세션 시작보다 window가 살짝 이르면(클라이언트 시계 오차 등), 가장 이른 window를
    기준점으로 삼아 음수 구간 없이 타임라인이 계산되어야 한다. #125에서 세션 시간 범위를
    크게 벗어난 window는 저장 자체가 안 되도록 필터링이 추가되었으므로, 필터 허용 오차
    (5분) 이내로 조정해서 검증한다."""
    _, headers = signup_and_login(client)
    session_id = _create_session(
        client,
        headers,
        tags=("충격 과다 러닝",),
        window_start_time=_STARTED_AT - 3 * 60 * 1000,
        window_end_time=_STARTED_AT - 1 * 60 * 1000,
    )

    monkeypatch.setattr(feedback_service, "_call_llm", lambda prompt: _MOCK_FEEDBACK)

    res = client.post(f"/running/{session_id}/feedback", headers=headers)

    assert res.status_code == 200
    assert res.json()["data"]["timeline"] == [
        {"tag": "충격 과다 러닝", "startMin": 0, "endMin": 2}
    ]


def test_feedback_timeline_batches_windows_by_majority_tag(client, monkeypatch):
    """태그가 바뀔 때마다 나뉘는 세부 구간 5개를 하나로 묶을 때, 그 안에서 가장 많이 나온
    태그가 대표로 나와야 한다. 태그를 매 window마다 번갈아 가며 보내 세부 구간이 병합되지
    않고 5개 그대로 배치에 들어가게 한다(3:2로 실제 다수결이 일어나는지 검증)."""
    _, headers = signup_and_login(client)
    session_id = _create_session_with_windows(
        client,
        headers,
        [("발끌기 러닝",), ("충격 과다 러닝",), ("발끌기 러닝",), ("충격 과다 러닝",), ("발끌기 러닝",)],
    )

    monkeypatch.setattr(feedback_service, "_call_llm", lambda prompt: _MOCK_FEEDBACK)

    res = client.post(f"/running/{session_id}/feedback", headers=headers)

    assert res.status_code == 200
    assert res.json()["data"]["timeline"] == [
        {"tag": "발끌기 러닝", "startMin": 0, "endMin": 2}
    ]


def test_feedback_timeline_majority_tie_prefers_stable_tag(client, monkeypatch):
    """다수결이 동점이면 _dominant_tag와 동일한 우선순위(안정 러닝 우선)로 하나를 정한다."""
    _, headers = signup_and_login(client)
    session_id = _create_session_with_windows(
        client, headers, [("안정 러닝",)] * 2 + [("충격 과다 러닝",)] * 2
    )

    monkeypatch.setattr(feedback_service, "_call_llm", lambda prompt: _MOCK_FEEDBACK)

    res = client.post(f"/running/{session_id}/feedback", headers=headers)

    assert res.status_code == 200
    assert res.json()["data"]["timeline"] == [
        {"tag": "안정 러닝", "startMin": 0, "endMin": 1}
    ]


def test_feedback_timeline_handles_remainder_batch(client, monkeypatch):
    """세부 구간이 5의 배수가 아니어도(나머지 배치) 다수결이 정상 동작하고, 앞 배치와
    대표 태그가 다르면 별도 구간으로 분리되어야 한다. 태그를 매 window마다 번갈아 가며
    보내 세부 구간 7개(5개짜리 배치 1개 + 2개짜리 나머지 배치 1개)로 나뉘게 한다."""
    _, headers = signup_and_login(client)
    session_id = _create_session_with_windows(
        client,
        headers,
        [
            ("발끌기 러닝",), ("충격 과다 러닝",), ("발끌기 러닝",), ("충격 과다 러닝",), ("발끌기 러닝",),
            ("충격 과다 러닝",), ("발끌기 러닝",),
        ],
    )

    monkeypatch.setattr(feedback_service, "_call_llm", lambda prompt: _MOCK_FEEDBACK)

    res = client.post(f"/running/{session_id}/feedback", headers=headers)

    assert res.status_code == 200
    assert res.json()["data"]["timeline"] == [
        {"tag": "발끌기 러닝", "startMin": 0, "endMin": 2},
        {"tag": "충격 과다 러닝", "startMin": 2, "endMin": 2},
    ]


def test_feedback_is_cached_on_second_call(client, monkeypatch):
    """force_regenerate 없이 두 번 호출하면, 두 번째는 캐시만 읽고 LLM을 다시 호출하지 않아야 한다."""
    _, headers = signup_and_login(client)
    session_id = _create_session(client, headers)

    call_count = {"n": 0}

    def _fake_call_llm(prompt):
        call_count["n"] += 1
        return _MOCK_FEEDBACK

    monkeypatch.setattr(feedback_service, "_call_llm", _fake_call_llm)

    first = client.post(f"/running/{session_id}/feedback", headers=headers)
    second = client.post(f"/running/{session_id}/feedback", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert call_count["n"] == 1
    assert first.json()["data"]["feedbackId"] == second.json()["data"]["feedbackId"]


def test_get_feedback_report_before_generation_is_404(client):
    _, headers = signup_and_login(client)
    session_id = _create_session(client, headers)

    res = client.get(f"/running/{session_id}/feedback", headers=headers)

    assert res.status_code == 404


def test_get_feedback_report_after_generation_does_not_call_llm(client, monkeypatch):
    """GET은 순수 조회 전용이라, 이미 생성된 피드백이 있으면 LLM을 절대 호출하면 안 된다."""
    _, headers = signup_and_login(client)
    session_id = _create_session(client, headers)

    call_count = {"n": 0}

    def _fake_call_llm(prompt):
        call_count["n"] += 1
        return _MOCK_FEEDBACK

    monkeypatch.setattr(feedback_service, "_call_llm", _fake_call_llm)
    client.post(f"/running/{session_id}/feedback", headers=headers)
    assert call_count["n"] == 1

    res = client.get(f"/running/{session_id}/feedback", headers=headers)

    assert res.status_code == 200
    assert res.json()["data"]["finalFeedback"]["summary"] == _MOCK_FEEDBACK.summary
    assert call_count["n"] == 1  # GET 호출로는 추가 호출이 없어야 함


def test_feedback_forbidden_for_other_users_session(client, monkeypatch):
    _, owner_headers = signup_and_login(client)
    session_id = _create_session(client, owner_headers)

    _, other_headers = signup_and_login(client)
    res = client.post(f"/running/{session_id}/feedback", headers=other_headers)

    assert res.status_code == 403


def test_feedback_session_not_found(client):
    _, headers = signup_and_login(client)

    res = client.post("/running/999999999/feedback", headers=headers)

    assert res.status_code == 404
