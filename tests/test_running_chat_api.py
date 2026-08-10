import pytest

from app.core.time import kst_naive_to_epoch_ms, now_kst_naive
from app.services import chat as chat_service
from app.services import feedback as feedback_service
from app.services import paper_rag
from tests.conftest import signup_and_login
from tests.test_running_api import _create_session

_MOCK_FEEDBACK = feedback_service._FeedbackOutput(
    summary="요약",
    strengths=["강점"],
    improvements=["개선점"],
    next_step="다음 스텝",
)

_MOCK_CHAT_OUTPUT = chat_service._ChatOutput(
    answer="이것은 모킹된 코칭 답변입니다.",
    used_source_ids=[1],
)


def test_ask_chat_success(client, monkeypatch):
    # 1. 회원가입 및 로그인
    _, headers = signup_and_login(client)

    # 2. 러닝 세션 생성
    session = _create_session(client, headers, started_at_ms=kst_naive_to_epoch_ms(now_kst_naive()))
    session_id = session["sessionId"]

    # 3. 러닝 피드백 생성
    monkeypatch.setattr(feedback_service, "_call_llm", lambda prompt: _MOCK_FEEDBACK)
    gen_res = client.post(f"/running/{session_id}/feedback", headers=headers)
    assert gen_res.status_code == 200

    # 4. LLM 및 RAG 검색 모킹
    monkeypatch.setattr(chat_service, "_call_llm", lambda prompt: _MOCK_CHAT_OUTPUT)
    monkeypatch.setattr(
        paper_rag,
        "search",
        lambda db, query: [
            {"doc_id": "doc1", "title": "논문1", "page": 5, "content": "논문 내용 1", "score": 0.8}
        ],
    )

    # 5. POST /running/{session_id}/chat (질문하기)
    chat_payload = {"question": "정강이 무리가 덜 가도록 뛰는 방법이 있을까요?"}
    res = client.post(f"/running/{session_id}/chat", json=chat_payload, headers=headers)

    assert res.status_code == 200
    res_data = res.json()
    assert res_data["status"] == "success"
    assert "data" in res_data
    assert "messageId" in res_data["data"]
    assert res_data["data"]["answer"] == "이것은 모킹된 코칭 답변입니다."
    assert len(res_data["data"]["sources"]) == 1
    assert res_data["data"]["sources"][0]["title"] == "논문1"

    # 6. GET /running/{session_id}/chat (대화 내역 확인)
    hist_res = client.get(f"/running/{session_id}/chat", headers=headers)
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert len(hist_data["data"]["messages"]) == 2  # user 질문 + assistant 답변
    assert hist_data["data"]["messages"][0]["role"] == "user"
    assert hist_data["data"]["messages"][0]["content"] == chat_payload["question"]
    assert hist_data["data"]["messages"][1]["role"] == "assistant"
    assert hist_data["data"]["messages"][1]["content"] == "이것은 모킹된 코칭 답변입니다."

    # 7. DELETE /running/{session_id}/chat (대화 리셋)
    del_res = client.delete(f"/running/{session_id}/chat", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["data"]["deletedCount"] == 2

    # 대화 내역이 완전히 비었는지 확인
    hist_res2 = client.get(f"/running/{session_id}/chat", headers=headers)
    assert hist_res2.status_code == 200
    assert len(hist_res2.json()["data"]["messages"]) == 0


def test_ask_chat_session_not_found(client):
    _, headers = signup_and_login(client)
    chat_payload = {"question": "질문"}
    res = client.post("/running/99999/chat", json=chat_payload, headers=headers)
    assert res.status_code == 404
    assert res.json()["message"] == "러닝 세션을 찾을 수 없습니다."


def test_ask_chat_session_forbidden(client):
    # 소유자 세션 생성
    _, owner_headers = signup_and_login(client)
    session = _create_session(client, owner_headers, started_at_ms=kst_naive_to_epoch_ms(now_kst_naive()))
    session_id = session["sessionId"]

    # 다른 유저로 로그인 후 접근 시도
    _, other_headers = signup_and_login(client)
    chat_payload = {"question": "질문"}
    res = client.post(f"/running/{session_id}/chat", json=chat_payload, headers=other_headers)
    assert res.status_code == 403
    assert res.json()["message"] == "본인의 세션이 아닙니다."


def test_ask_chat_feedback_not_found(client):
    _, headers = signup_and_login(client)
    session = _create_session(client, headers, started_at_ms=kst_naive_to_epoch_ms(now_kst_naive()))
    session_id = session["sessionId"]

    # 피드백을 생성하지 않은 상태에서 질문 시도
    chat_payload = {"question": "질문"}
    res = client.post(f"/running/{session_id}/chat", json=chat_payload, headers=headers)
    assert res.status_code == 404
    assert res.json()["message"] == "생성된 피드백이 없습니다."
