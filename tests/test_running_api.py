from app.core.time import epoch_ms_to_kst_naive, kst_naive_to_epoch_ms, now_kst_naive
from app.services import feedback as feedback_service
from tests.conftest import signup_and_login

_WEEKDAY_LABELS = ["월", "화", "수", "목", "금", "토", "일"]

_MOCK_FEEDBACK = feedback_service._FeedbackOutput(
    summary="요약",
    strengths=["강점"],
    improvements=["개선점"],
    next_step="다음 스텝",
)


def _session_payload(started_at_ms, duration_sec=300, distance_meters=1000.0, tags=("안정 러닝",)):
    ended_at_ms = started_at_ms + duration_sec * 1000
    return {
        "mode": "outdoor",
        "route": None,
        "startedAt": started_at_ms,
        "endedAt": ended_at_ms,
        "durationSec": duration_sec,
        "distanceMeters": distance_meters,
        "windows": [
            {
                "predictions": ["normal"],
                "tags": list(tags),
                "feedbackText": None,
                "dtwContext": None,
                "windowStartTime": started_at_ms,
                "windowEndTime": started_at_ms + 10_000,
            }
        ],
    }


def _create_session(client, headers, **kwargs):
    payload = _session_payload(**kwargs)
    res = client.post("/running/complete", json=payload, headers=headers)
    assert res.status_code == 200
    return res.json()["data"]


# ---------------------------------------------------------------------------
# POST /running/complete
# ---------------------------------------------------------------------------


def test_complete_running_saves_kst_converted_time(client):
    _, headers = signup_and_login(client)
    started_at_ms = kst_naive_to_epoch_ms(now_kst_naive())

    data = _create_session(client, headers, started_at_ms=started_at_ms, distance_meters=1234.0)

    expected_start = epoch_ms_to_kst_naive(started_at_ms)
    assert data["startTime"].replace("T", " ")[:19] == expected_start.strftime("%Y-%m-%d %H:%M:%S")
    assert float(data["totalDistanceM"]) == 1234.0


def test_complete_running_requires_token(client):
    res = client.post("/running/complete", json=_session_payload(started_at_ms=1_700_000_000_000))

    assert res.status_code == 401


def test_complete_running_filters_windows_outside_session_range(client):
    """세션 시간대와 무관한(예: 이전 녹화 버퍼) window는 저장에서 제외되어야 한다."""
    _, headers = signup_and_login(client)
    started_at_ms = kst_naive_to_epoch_ms(now_kst_naive())
    payload = _session_payload(started_at_ms=started_at_ms)
    stale_window_start = started_at_ms - 90 * 60_000  # 세션 시작보다 90분 이전
    payload["windows"].append(
        {
            "predictions": ["normal"],
            "tags": ["발끌기 러닝"],
            "feedbackText": None,
            "dtwContext": None,
            "windowStartTime": stale_window_start,
            "windowEndTime": stale_window_start + 10_000,
        }
    )

    res = client.post("/running/complete", json=payload, headers=headers)

    assert res.status_code == 200
    windows = res.json()["data"]["windows"]
    assert len(windows) == 1
    assert windows[0]["windowStartTime"] == started_at_ms


def test_complete_running_keeps_windows_within_tolerance(client):
    """클라이언트 시계 오차 수준(5분 이내)의 window는 정상적으로 저장되어야 한다."""
    _, headers = signup_and_login(client)
    started_at_ms = kst_naive_to_epoch_ms(now_kst_naive())
    payload = _session_payload(started_at_ms=started_at_ms)
    near_window_start = started_at_ms - 4 * 60_000  # 세션 시작보다 4분 이전 (허용 오차 이내)
    payload["windows"].append(
        {
            "predictions": ["normal"],
            "tags": ["안정 러닝"],
            "feedbackText": None,
            "dtwContext": None,
            "windowStartTime": near_window_start,
            "windowEndTime": near_window_start + 10_000,
        }
    )

    res = client.post("/running/complete", json=payload, headers=headers)

    assert res.status_code == 200
    windows = res.json()["data"]["windows"]
    assert len(windows) == 2


# ---------------------------------------------------------------------------
# GET /running/home
# ---------------------------------------------------------------------------


def test_home_summary_aggregates_todays_session(client):
    _, headers = signup_and_login(client)
    today = now_kst_naive()
    started_at_ms = kst_naive_to_epoch_ms(today)
    _create_session(client, headers, started_at_ms=started_at_ms, distance_meters=1000.0)

    res = client.get("/running/home", headers=headers)

    assert res.status_code == 200
    data = res.json()["data"]
    assert len(data["dailyDistances"]) == 7
    assert [d["weekdayLabel"] for d in data["dailyDistances"]] == _WEEKDAY_LABELS

    todays_entry = data["dailyDistances"][today.weekday()]
    assert todays_entry["distanceKm"] == 1.0
    assert data["weekTotalKm"] == 1.0


def test_home_summary_empty_when_no_sessions(client):
    _, headers = signup_and_login(client)

    res = client.get("/running/home", headers=headers)

    assert res.status_code == 200
    data = res.json()["data"]
    assert data["weekTotalKm"] == 0
    assert all(d["distanceKm"] == 0 for d in data["dailyDistances"])


def test_home_summary_requires_token(client):
    res = client.get("/running/home")

    assert res.status_code == 401


# ---------------------------------------------------------------------------
# GET /running/monthly-summary
# ---------------------------------------------------------------------------


def test_monthly_summary_counts_this_months_session(client):
    _, headers = signup_and_login(client)
    today = now_kst_naive()
    started_at_ms = kst_naive_to_epoch_ms(today)
    session = _create_session(client, headers, started_at_ms=started_at_ms, distance_meters=2000.0)

    res = client.get(
        "/running/monthly-summary", params={"year": today.year, "month": today.month}, headers=headers
    )

    assert res.status_code == 200
    data = res.json()["data"]
    assert data["totalCount"] == 1
    assert data["totalDistanceKm"] == 2.0
    assert data["recentSessions"][0]["sessionId"] == session["sessionId"]
    # 피드백을 아직 생성 안 했으니 대표 태그도, 전체 태그 분포도 비어있어야 한다.
    assert data["recentSessions"][0]["dominantTag"] is None
    assert data["tagRatios"] == []


def test_monthly_summary_empty_for_month_with_no_sessions(client):
    _, headers = signup_and_login(client)

    res = client.get("/running/monthly-summary", params={"year": 2000, "month": 1}, headers=headers)

    assert res.status_code == 200
    data = res.json()["data"]
    assert data["totalCount"] == 0
    assert data["totalDistanceKm"] == 0
    assert data["recentSessions"] == []
    assert data["tagRatios"] == []


def test_monthly_summary_invalid_month_rejected(client):
    _, headers = signup_and_login(client)

    res = client.get("/running/monthly-summary", params={"year": 2026, "month": 13}, headers=headers)

    assert res.status_code == 422


# ---------------------------------------------------------------------------
# GET /running/{session_id}/feedback/summary
# ---------------------------------------------------------------------------


def test_feedback_summary_returns_tag_ratio_and_timeline(client, monkeypatch):
    _, headers = signup_and_login(client)
    session = _create_session(client, headers, started_at_ms=kst_naive_to_epoch_ms(now_kst_naive()))
    session_id = session["sessionId"]

    monkeypatch.setattr(feedback_service, "_call_llm", lambda prompt: _MOCK_FEEDBACK)
    gen_res = client.post(f"/running/{session_id}/feedback", headers=headers)
    assert gen_res.status_code == 200

    res = client.get(f"/running/{session_id}/feedback/summary", headers=headers)

    assert res.status_code == 200
    data = res.json()["data"]
    assert data["tagRatio"] == {"안정 러닝": 1.0}
    assert isinstance(data["timeline"], list)


def test_feedback_summary_before_generation_is_404(client):
    _, headers = signup_and_login(client)
    session = _create_session(client, headers, started_at_ms=kst_naive_to_epoch_ms(now_kst_naive()))

    res = client.get(f"/running/{session['sessionId']}/feedback/summary", headers=headers)

    assert res.status_code == 404


def test_feedback_summary_forbidden_for_other_users_session(client):
    _, owner_headers = signup_and_login(client)
    session = _create_session(
        client, owner_headers, started_at_ms=kst_naive_to_epoch_ms(now_kst_naive())
    )

    _, other_headers = signup_and_login(client)
    res = client.get(f"/running/{session['sessionId']}/feedback/summary", headers=other_headers)

    assert res.status_code == 403
