import json
import logging
from collections import Counter
from statistics import mean

from google import genai
from langsmith import traceable
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.crud.running import get_feedback_by_session, get_windows_by_session, upsert_feedback
from app.models.running import SessionWindow
from app.schemas.running import RunningFeedbackOut
from app.services.knowledge import get_kb, pick_docs, validate_sources

logger = logging.getLogger(__name__)

FALLBACK_TEXT = "이번 러닝 세션을 분석할 수 없었습니다. 다음 러닝 후 다시 시도해주세요."


class _FeedbackOutput(BaseModel):
    text: str
    source_ids: list[str]


def select_dominant_tags(tag_ratio: dict[str, float], threshold: float = 0.20, max_tags: int = 3) -> list[str]:
    ratio = {t: r for t, r in tag_ratio.items() if t and t != "안정 러닝"}
    picked = sorted([(t, r) for t, r in ratio.items() if r >= threshold], key=lambda x: -x[1])[:max_tags]
    return [t for t, _ in picked]


@traceable(name="summarize_dtw")
def summarize_dtw(windows: list[SessionWindow]) -> dict:
    seq = [(w.window_start_time, w.dtw_context) for w in windows if w.dtw_context and "avgSimilarity" in w.dtw_context]
    if not seq:
        return {"trend": "no_data"}
    sims = [ctx["avgSimilarity"] for _, ctx in seq]
    third = max(len(sims) // 3, 1)
    start_avg, end_avg = round(mean(sims[:third]), 2), round(mean(sims[-third:]), 2)
    peak_idx, trough_idx = sims.index(max(sims)), sims.index(min(sims))
    return {
        "start_avg": start_avg,
        "end_avg": end_avg,
        "trend": "increasing" if end_avg > start_avg else "decreasing" if end_avg < start_avg else "stable",
        "peak_similarity": round(sims[peak_idx], 2),
        "trough_similarity": round(sims[trough_idx], 2),
    }


@traceable(name="resolve_docs")
def resolve_docs(tag_ratio: dict, dominant_tags: list[str], kb: dict) -> tuple[list[dict], str]:
    if tag_ratio.get("안정 러닝", 0) >= 0.7 or not dominant_tags:
        return pick_docs("안정 러닝", kb), "encouragement"
    docs = [d for t in dominant_tags for d in pick_docs(t, kb)]
    if not docs:
        return pick_docs("안정 러닝", kb), "encouragement"
    return docs, "corrective"


def _coaching_history(windows: list[SessionWindow]) -> list[dict]:
    result = []
    for w in windows:
        if w.feedback_text and w.tags:
            result.append({
                "tags": w.tags,
                "feedback": w.feedback_text,
                "window_start": w.window_start_time,
            })
    return result


def _build_feedback_prompt(
    session_id: int,
    tag_ratio: dict,
    dominant_tags: list[str],
    dtw_summary: dict,
    docs: list[dict],
    prompt_mode: str,
    coaching_history: list[dict] | None = None,
) -> str:
    lines = [
        "[러닝 세션 요약]",
        f"- 세션 ID: {session_id}",
        f"- 태그 비율: {json.dumps(tag_ratio, ensure_ascii=False)}",
        f"- 주요 태그: {', '.join(dominant_tags) if dominant_tags else '없음'}",
        f"- 모드: {prompt_mode}",
        "",
        "[DTW 유사도 추이]",
        f"- 전반부 평균: {dtw_summary.get('start_avg', 'N/A')}",
        f"- 후반부 평균: {dtw_summary.get('end_avg', 'N/A')}",
        f"- 추세: {dtw_summary.get('trend', 'N/A')}",
        f"- 최고 유사도: {dtw_summary.get('peak_similarity', 'N/A')}",
        f"- 최저 유사도: {dtw_summary.get('trough_similarity', 'N/A')}",
        "",
        "[참고 문헌]",
        *[
            f"- {d['source']} ({d['tag']} / {d['type']}): {d['text']}"
            for d in docs
        ],
    ]
    if coaching_history:
        lines += [
            "",
            "[구간별 코칭 피드백 (이미 사용자에게 전달됨)]",
            *[
                f"- 태그: {', '.join(h['tags'])} → {h['feedback']}"
                for h in coaching_history
            ],
            "",
            "※ 위 구간별 피드백은 이미 러닝 중에 사용자에게 전달된 내용입니다. "
            "이를 참고하여 같은 말을 반복하지 말고, 세션 전체를 관통하는 종합적인 인사이트를 새로 제공하세요.",
        ]
    lines += [
        "",
        "위 정보를 바탕으로 사용자에게 러닝 세션 전체에 대한 종합 피드백을 한국어로 3~5문장 이내로 작성해주세요.",
        "출처를 인용할 때는 문헌 ID(예: impact_drill_sara2024)를 source_ids에 포함시켜 주세요.",
    ]
    return "\n".join(lines)


_SYSTEM_INSTRUCTION = (
    "당신은 러닝 코칭 전문가입니다. 제공된 세션 태그 비율, DTW 유사도 추이, 참고 문헌을 종합하여 "
    "사용자에게 과학적이고 구체적인 피드백을 한국어로 제공합니다.\n\n"
    "응답은 반드시 JSON 형식으로 다음 필드를 포함해야 합니다:\n"
    "  - text: 3~5문장의 종합 피드백 (한국어)\n"
    "  - source_ids: 피드백에서 인용한 참고 문헌의 ID 목록\n\n"
    "prompt_mode가 'encouragement'이면 칭찬과 유지 방법 위주로, "
    "'corrective'이면 개선점과 구체적인 교정 방법 위주로 작성하세요.\n\n"
    "※ '구간별 코칭 피드백'은 러닝 중 이미 사용자에게 전달된 조언입니다. "
    "같은 내용을 반복하지 말고, 세션 전체를 관통하는 상위 인사이트를 새로 제공하세요. "
    '예를 들어 3km 지점에서 케이던스 피드백을 드렸다면, 이는 후반부로 갈수록 보폭이 좁아지는 전체적인 패턴의 일부였다고 설명하는 식으로 구체적 피드백을 맥락화해주세요.'
)


@traceable(name="call_llm")
def _call_llm(prompt: str) -> tuple[str, list[str]] | None:
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=4000,
                response_mime_type="application/json",
                response_schema=_FeedbackOutput,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None and response.text:
            parsed = _FeedbackOutput(**json.loads(response.text))
        if parsed is None or not parsed.text:
            return None
        return parsed.text.strip(), parsed.source_ids
    except Exception as e:
        logger.error("LLM 호출 실패 (세션 피드백): %s", e)
        return None


@traceable(name="generate_session_feedback", project_name="azas-feedback")
def generate_session_feedback(
    db: Session, session_id: int, user_id, force_regenerate: bool = False
) -> RunningFeedbackOut:
    existing = get_feedback_by_session(db, session_id)
    if existing and not force_regenerate:
        return RunningFeedbackOut.model_validate(existing)

    windows = get_windows_by_session(db, session_id)

    all_tags = [t for w in windows for t in (w.tags or [])]
    total = len(all_tags) or 1
    tag_ratio = {t: c / total for t, c in Counter(all_tags).items()}

    dominant_tags = select_dominant_tags(tag_ratio)
    dtw_summary = summarize_dtw(windows)
    coaching_history = _coaching_history(windows)
    kb = get_kb()
    docs, prompt_mode = resolve_docs(tag_ratio, dominant_tags, kb)
    prompt = _build_feedback_prompt(
        session_id, tag_ratio, dominant_tags, dtw_summary, docs, prompt_mode, coaching_history,
    )

    llm_result = _call_llm(prompt)
    if llm_result is None:
        data = {"final_feedback": FALLBACK_TEXT, "tag_ratio": tag_ratio}
        fb = upsert_feedback(db, session_id, user_id, data)
        return RunningFeedbackOut.model_validate(fb)

    text, source_ids = llm_result
    validated = validate_sources(source_ids, kb)
    if source_ids:
        logger.debug("LLM source_ids=%s → validated=%s", source_ids, [v["id"] for v in validated])
    data = {"final_feedback": text, "tag_ratio": tag_ratio}
    fb = upsert_feedback(db, session_id, user_id, data)
    return RunningFeedbackOut.model_validate(fb)
