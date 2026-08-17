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


def _create_session(client, headers, tags=("안정 러닝",)):
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
                "windowStartTime": _STARTED_AT,
                "windowEndTime": _STARTED_AT + 10_000,
            }
        ],
    }
    res = client.post("/running/complete", json=payload, headers=headers)
    assert res.status_code == 200
    return res.json()["data"]["sessionId"]


def test_batch_majority_tag_prefers_longer_duration_on_tie():
    """개수가 동점이면 우선순위가 아니라 묶음 안에서 차지한 총 길이가 더 긴 태그를 고른다."""
    chunk = [
        {"tag": "안정 러닝", "start_ms": 0, "end_ms": 60_000},
        {"tag": "충격 과다 러닝", "start_ms": 60_000, "end_ms": 240_000},
    ]

    assert feedback_service._batch_majority_tag(chunk) == "충격 과다 러닝"


def test_batch_majority_tag_returns_sole_winner_without_tie():
    """개수 자체가 동점이 아니면 길이 계산 없이 최다 득표 태그를 그대로 고른다."""
    chunk = [
        {"tag": "발끌기 러닝", "start_ms": 0, "end_ms": 10_000},
        {"tag": "발끌기 러닝", "start_ms": 10_000, "end_ms": 20_000},
        {"tag": "충격 과다 러닝", "start_ms": 20_000, "end_ms": 500_000},
    ]

    assert feedback_service._batch_majority_tag(chunk) == "발끌기 러닝"


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
