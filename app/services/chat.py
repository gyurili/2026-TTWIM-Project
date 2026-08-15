"""러닝 최종 리포트 화면의 멀티턴 RAG 챗봇 서비스.

세션의 실제 수치(feedback.py가 이미 만들어 둔 요약/프로필/문헌 선택 로직을 재사용)와
app/services/paper_rag.py가 검색한 논문 근거를 함께 Gemini에 넣어 후속 질문에 답한다.
"""

import json
import logging
import random
import re
import time
from uuid import UUID

from google import genai
from google.genai import types
from langsmith import traceable
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import FeedbackNotFoundError
from app.crud.chat import create_messages, get_messages_by_session
from app.crud.running import get_feedback_by_session
from app.models.chat import ChatMessage
from app.models.running import RunningSession
from app.schemas.chat import ChatAnswerOut, ChatSourceOut
from app.schemas.running import FeedbackDetail
from app.services import paper_rag
from app.services.feedback import _build_user_profile, _summarize_session, resolve_docs
from app.services.knowledge import get_kb

logger = logging.getLogger(__name__)

_ANSWER_MODEL = "gemini-3.1-flash-lite"
_REWRITE_MODEL = "gemini-3.1-flash-lite"
_HISTORY_TURNS = 6  # user+assistant 6쌍 = 최근 12메시지
_SNIPPET_MAX_CHARS = 120

_LLM_TIMEOUT_MS = 30_000
_LLM_MAX_RETRIES = 1
_LLM_RETRY_BACKOFF_BASE_SEC = 1.5
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

_QUERY_REWRITE_ENABLED = True  # scripts/eval_retrieval.py로 검증된 기본 유지 값. 튜닝 대상 아님

_FALLBACK_ANSWER = "일시적인 오류로 답변을 생성하지 못했습니다. 잠시 후 다시 시도해주세요."

# 시스템 프롬프트로 인용 마커를 넣지 말라고 지시하지만, 모델이 지시를 놓칠 때를 대비한 안전장치.
_CITATION_MARKER_RE = re.compile(r"\s*\[S\d+\]")


def _strip_citation_markers(text: str) -> str:
    return _CITATION_MARKER_RE.sub("", text).strip()

_SYSTEM_INSTRUCTION = (
    "당신은 러닝 코칭 전문가입니다. 사용자는 방금 끝난 러닝 세션의 최종 리포트를 보고 있으며, "
    "그에 대한 후속 질문을 합니다. 아래 제공된 정보만 근거로 답하세요.\n\n"
    "- 한국어로 간결하게 답하세요. 문장을 길게 늘여 쓰지 말고 짧게 끊어 쓰며, 전체 답변은 "
    "4문장을 넘기지 마세요.\n"
    "- 원인 설명, 조언, 주의사항처럼 서로 다른 이야기가 섞이면 한 문단에 다 욱여넣지 말고 "
    "1~2문장 단위로 문단을 나눠 그 사이에 빈 줄(\\n\\n)을 넣으세요. 답변 전체가 줄바꿈 없는 "
    "하나의 긴 덩어리 문단이 되지 않게 하세요.\n"
    "- 실행할 방법이나 팁이 2개 이상 나열될 때는 문장으로 풀어쓰지 말고 마크다운 리스트로 "
    "쓰세요. 순서가 있으면 '1. 2. 3.', 순서가 없으면 '- '를 각 줄 맨 앞에 붙이고, 각 줄은 "
    "새 줄(\\n)로 구분하세요. 리스트 항목 하나는 한 문장 이내로 짧게 쓰고, 리스트는 최대 4개 "
    "항목까지만 쓰세요. 이 경우 위의 '4문장 이내' 기준은 리스트 항목이 아니라 리스트 앞뒤의 "
    "설명 문장에만 적용됩니다.\n"
    "- 답변에서 사용자가 실제로 기억하고 실행해야 할 핵심 키워드(구체적인 동작·부위)와 "
    "목표 수치(퍼센트, 횟수, 시간 등)는 반드시 마크다운 굵게 표시(**단어**)로 감싸세요. "
    "예: **케이던스**를 **5~10%** 높여보세요. 문단마다 많아야 2~3개만 강조하고, "
    "문장 전체나 조사(을/를/이/가 등)까지 함께 감싸지 마세요.\n"
    "- [러닝 세션 요약], [생성된 최종 리포트], [구간별 타임라인]에 있는 수치만 사용하고 "
    "그 안에 없는 수치나 사실을 지어내지 마세요.\n"
    "- 옆에서 함께 뛰는 코치가 말로 설명해주듯 쉽고 자연스러운 구어체로 쓰세요. [검색된 논문 근거]의 "
    "문장을 그대로 옮기거나 학술적인 어투로 답하지 말고, 전문 용어는 일상적인 말로 풀어서 설명하세요. "
    "예를 들어 '경골 스트레스', '수직 부하율 증가' 같은 표현 대신 '정강이에 무리가 갈 수 있어요', "
    "'착지할 때 충격이 세게 실려요'처럼 쓰세요.\n"
    "- 답변 문장 안에는 [S1]처럼 근거 번호나 출처를 나타내는 표시를 절대 넣지 마세요. 어떤 근거를 "
    "사용했는지는 answer 텍스트가 아니라 used_source_ids 필드로만 전달합니다.\n"
    "- 근거가 없거나 부족하면 '논문에서 확인된 내용은 아니지만'이라고 먼저 밝히고 일반적인 러닝 지식으로 답하세요.\n"
    "- 러닝, 러닝 자세, 훈련과 무관한 질문(예: 일상 대화, 다른 주제)은 정중히 답변을 거절하고 "
    "러닝 관련 질문을 유도하세요.\n"
    "- 의학적 진단이나 처방은 하지 마세요. 통증이나 부상이 의심되는 내용이 있으면 전문의 상담을 권유하세요.\n\n"
    "used_source_ids를 정할 때는 [검색된 논문 근거] 번호 하나하나를 개별적으로 판단하세요. "
    "'같은 문서에서 나왔다', '주제가 비슷하다'는 이유만으로는 포함하지 마세요. 오직 answer 안의 "
    "특정 문장이 '그 번호의 근거 내용이 없었다면 쓸 수 없었을 정도로' 직접 뒷받침될 때만 포함하세요. "
    "예를 들어 근거가 [S1] 케이던스를 높이는 방법, [S2] 러닝머신에서 영상 찍어 자세 확인하는 방법을 "
    "다루는데 answer에서 케이던스 얘기만 했다면 used_source_ids는 [1]만 넣고 2는 빼야 합니다 — "
    "주제가 같은 문서에 있다고 해서 함께 넣지 마세요.\n\n"
    "응답은 JSON 스키마(answer, used_source_ids)를 따르세요. answer에는 인용 표시를 넣지 말고, "
    "위 기준을 통과한 근거 번호(정수, [S1]이면 1)만 used_source_ids에 넣으세요. "
    "그런 근거가 하나도 없다면 빈 리스트로 두세요."
)


class _ChatOutput(BaseModel):
    answer: str
    used_source_ids: list[int] = Field(default_factory=list)


class _RewriteOutput(BaseModel):
    standalone_question: str


def _is_retryable_error(e: Exception) -> bool:
    status = getattr(e, "code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES
    name = type(e).__name__.lower()
    return any(keyword in name for keyword in ("timeout", "connection", "deadline"))


def _dominant_tag_label(tag_ratio: dict) -> str | None:
    if not tag_ratio:
        return None
    return max(tag_ratio.items(), key=lambda kv: kv[1])[0]


@traceable(name="rewrite_chat_query")
def _rewrite_query(client: genai.Client, question: str, history: list[ChatMessage]) -> str:
    """대화 이력이 있을 때 대명사/생략이 섞인 질문을 검색 가능한 독립 질문으로 재작성한다.
    실패하면 직전 user 질문과 단순 결합해 폴백한다 (검색 완전 실패보다 낫다)."""
    history_text = "\n".join(f"{'사용자' if m.role == 'user' else '챗봇'}: {m.content}" for m in history)
    prompt = (
        f"[이전 대화]\n{history_text}\n\n"
        f"[현재 질문]\n{question}\n\n"
        "현재 질문이 이전 대화의 맥락(대명사, 생략된 주어 등)에 의존한다면, 이전 대화 없이도 "
        "이해할 수 있는 독립적인 질문 한 문장으로 바꿔주세요. 현재 질문이 이미 그 자체로 "
        "완결되어 있다면 그대로 반환하세요. 질문의 언어(한국어)는 유지하세요."
    )
    try:
        response = client.models.generate_content(
            model=_REWRITE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=200,
                response_mime_type="application/json",
                response_schema=_RewriteOutput,
                http_options=types.HttpOptions(timeout=10_000),
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None and response.text:
            parsed = _RewriteOutput.model_validate_json(response.text)
        if parsed and parsed.standalone_question.strip():
            return parsed.standalone_question.strip()
    except Exception as e:
        logger.warning("검색 쿼리 재작성 실패, 문자열 결합으로 폴백: %s", e)

    last_user = next((m.content for m in reversed(history) if m.role == "user"), "")
    return f"{last_user} {question}".strip()


def _build_search_query(client: genai.Client, question: str, history: list[ChatMessage], dominant_tag: str | None) -> str:
    if history and _QUERY_REWRITE_ENABLED:
        query = _rewrite_query(client, question, history)
    else:
        query = question
    return f"{query} ({dominant_tag})" if dominant_tag else query


def _format_history(history: list[ChatMessage]) -> str:
    if not history:
        return "(이전 대화 없음)"
    return "\n".join(f"{'사용자' if m.role == 'user' else '챗봇'}: {m.content}" for m in history)


def _build_chat_prompt(
    question: str,
    session_summary: dict,
    final_feedback: dict,
    timeline: list[dict],
    paper_sources: list[dict],
    kb_docs: list[dict],
    user_profile: dict,
    history: list[ChatMessage],
) -> str:
    lines = [
        "[사용자 프로필]",
        f"- 나이: {user_profile['age']}세",
        f"- 성별: {user_profile['gender']}",
        f"- 키: {user_profile['height']}cm",
        f"- 몸무게: {user_profile['weight']}kg",
        "",
        "[이번 러닝 요약]",
        f"- 거리: {session_summary['distance_km']}km",
        f"- 시간: {session_summary['duration_display']}",
        f"- 평균 페이스: {session_summary['pace_display']}",
        f"- 러닝 종류: {session_summary['mode_label']}",
        "",
        "[생성된 최종 리포트]",
        f"- 요약: {final_feedback.get('summary', '')}",
        f"- 잘한 점: {'; '.join(final_feedback.get('strengths', []))}",
        f"- 개선점: {'; '.join(final_feedback.get('improvements', []))}",
        f"- 다음 단계: {final_feedback.get('next_step', '')}",
    ]

    if timeline:
        lines += [
            "",
            "[구간별 타임라인]",
            *[f"- {t['start_min']}~{t['end_min']}분: {t['tag']}" for t in timeline],
        ]

    lines += ["", "[검색된 논문 근거]"]
    if paper_sources:
        lines += [f"- [S{i}] {s['title']} p.{s['page']} — {s['content']}" for i, s in enumerate(paper_sources, start=1)]
    else:
        lines += ["- (검색된 논문 근거 없음)"]

    if kb_docs:
        lines += [
            "",
            "[참고 요약 자료]",
            *[f"- {d['text']}" for d in kb_docs],
        ]

    lines += [
        "",
        "[이전 대화]",
        _format_history(history),
        "",
        "[사용자 질문]",
        question,
    ]
    return "\n".join(lines)


@traceable(name="call_chat_llm")
def _call_llm(prompt: str) -> _ChatOutput | None:
    last_error: Exception | None = None
    for attempt in range(_LLM_MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            response = client.models.generate_content(
                model=_ANSWER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    temperature=0.3,
                    max_output_tokens=1200,
                    response_mime_type="application/json",
                    response_schema=_ChatOutput,
                    http_options=types.HttpOptions(timeout=_LLM_TIMEOUT_MS),
                ),
            )
            parsed = getattr(response, "parsed", None)
            if parsed is None and response.text:
                parsed = _ChatOutput.model_validate_json(response.text)
            return parsed
        except Exception as e:
            last_error = e
            retryable = _is_retryable_error(e)
            logger.warning(
                "LLM 호출 실패 (챗봇, 시도 %d/%d, retryable=%s): %s",
                attempt + 1, _LLM_MAX_RETRIES + 1, retryable, e,
            )
            if not retryable or attempt == _LLM_MAX_RETRIES:
                break
            time.sleep(_LLM_RETRY_BACKOFF_BASE_SEC * (2**attempt) + random.uniform(0, 0.5))

    logger.error("LLM 호출 최종 실패 (챗봇): %s", last_error)
    return None


@traceable(name="answer_question", project_name="azas-chat")
def answer_question(db: Session, session: RunningSession, user, question: str) -> ChatAnswerOut:
    feedback = get_feedback_by_session(db, session.session_id)
    if not feedback:
        raise FeedbackNotFoundError()

    final_feedback = FeedbackDetail.model_validate(
        json.loads(feedback.final_feedback) if isinstance(feedback.final_feedback, str) else feedback.final_feedback
    ).model_dump()
    tag_ratio = feedback.tag_ratio or {}
    timeline = feedback.timeline or []

    history = get_messages_by_session(db, session.session_id, limit=_HISTORY_TURNS * 2)

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    dominant_tag = _dominant_tag_label(tag_ratio)
    search_query = _build_search_query(client, question, history, dominant_tag)

    paper_sources = paper_rag.search(db, search_query)
    kb_docs, _, _ = resolve_docs(tag_ratio, get_kb())

    session_summary = _summarize_session(session)
    user_profile = _build_user_profile(user)

    prompt = _build_chat_prompt(
        question=question,
        session_summary=session_summary,
        final_feedback=final_feedback,
        timeline=timeline,
        paper_sources=paper_sources,
        kb_docs=kb_docs,
        user_profile=user_profile,
        history=history,
    )

    result = _call_llm(prompt)

    if result is None:
        answer_text = _FALLBACK_ANSWER
        cited_sources: list[dict] = []
    else:
        answer_text = _strip_citation_markers(result.answer)
        cited_sources = [
            paper_sources[i - 1] for i in result.used_source_ids if 1 <= i <= len(paper_sources)
        ]

    source_payload = [
        {
            "docId": s["doc_id"],
            "title": s["title"],
            "page": s["page"],
            "snippet": s["content"][:_SNIPPET_MAX_CHARS],
        }
        for s in cited_sources
    ]

    saved = create_messages(
        db,
        session.session_id,
        user.id,
        [("user", question, None), ("assistant", answer_text, source_payload or None)],
    )
    assistant_msg = saved[-1]

    return ChatAnswerOut(
        message_id=assistant_msg.message_id,
        answer=answer_text,
        sources=[ChatSourceOut(doc_id=s["docId"], title=s["title"], page=s["page"], snippet=s["snippet"]) for s in source_payload],
    )
