import json
import logging
import random
import time
from collections import Counter
from datetime import date
from statistics import mean

from google import genai
from langsmith import traceable
from pydantic import BaseModel, Field
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.time import kst_naive_to_epoch_ms
from app.crud.running import get_feedback_by_session, get_windows_by_session, upsert_feedback
from app.models.running import RunningFeedback, RunningSession, SessionWindow
from app.schemas.running import RunningFeedbackOut
from app.services.knowledge import get_kb, pick_docs

logger = logging.getLogger(__name__)

FALLBACK_TEXT = "이번 러닝 세션을 분석할 수 없었습니다. 다음 러닝 후 다시 시도해주세요."
INSUFFICIENT_DATA_TEXT = "이번 러닝 세션은 분석할 수 있는 데이터가 부족합니다. 트래킹 상태를 확인한 뒤 다음 러닝에서 다시 시도해주세요."

_MAX_COACHING_HISTORY = 10
_DTW_SEGMENTS = 5
_MAX_HISTORY_FEEDBACK_CHARS = 80  # 구간별 피드백 원문이 길 때 프롬프트 비대화를 막기 위한 절삭 길이

# 타임라인 구간의 대표 태그를 정할 때 쓰는 우선순위. "안정 러닝"은 다른 태그와 동시에 나오지 않는
# 값이라 별도 취급하고(_dominant_tag 참고), 문제 태그가 여러 개인 window에서는 이 순서로 하나를 고른다.
_STABLE_TAG = "안정 러닝"
_ISSUE_TAG_PRIORITY = ["충격 과다 러닝", "발끌기 러닝", "좁은 보폭 러닝", "페이스 변화 러닝"]

# LLM 호출 타임아웃/재시도 설정.
_LLM_TIMEOUT_MS = 60_000
_LLM_MAX_RETRIES = 2  # 최초 시도 포함 총 3회 호출
_LLM_RETRY_BACKOFF_BASE_SEC = 1.5
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# 시도당 타임아웃만 두면 재시도가 겹칠 때 총 소요시간이 60초의 배수로 계속 늘어난다
# (예: 1차 시도가 60초 꽉 채워 타임아웃 → 2차 시도가 또 60초 근처까지 걸리면 총 2분 이상).
# 그래서 _call_llm 전체(모든 시도+백오프 합산)에 상한을 두고, 그 예산 안에서만 재시도한다.
# 프론트 axios 타임아웃(현재 1분)보다 몇 초 여유 있게 짧게 잡아서, 프론트가 타임아웃으로
# 판단하기 전에 백엔드가 먼저 (성공이든 폴백이든) 응답을 끝내도록 한다.
# 주의: 프론트 타임아웃 값이 바뀌면 이 값도 같이 재검토해야 한다.
_LLM_TOTAL_BUDGET_SEC = 55

# dtwContext 해석 가이드에 명시된 값. 임의로 정한 게 아니라 데이터 생성 쪽(ML/DTW) 스펙을 그대로 따른다.
_STATUS_LABELS = {"stable": "안정 (평소와 거의 같음)", "drift": "이탈 시작 (주의 관찰 필요)", "alert": "경고 (개입 권장)"}
_STATUS_SEVERITY = {"stable": 0, "drift": 1, "alert": 2}
_TREND_LABELS = {"improving": "회복 중", "steady": "유지", "worsening": "악화 중"}
_SHAPE_DEV_NOTABLE = 2.0     # 가이드: 대략 2 이상이면 뚜렷한 걸음 형태 변화
_CADENCE_DEV_NOTABLE = 3.0   # 가이드: 대략 ±3 이상이면 뚜렷한 속도 변화
# amplitudeDev는 가이드에 명시된 절대 임계값이 없음 -> "뚜렷하다"고 단정하지 않고 원값+방향만 제공한다.


class _FeedbackOutput(BaseModel):
    summary: str = Field(description="세션 흐름을 원인과 결과로 이어서 서술하는 최소 2문장 이상. 항목을 나열하지 말고 하나의 이야기처럼 자연스럽게 연결할 것. 핵심 키워드나 수치는 **볼드**로 강조할 것")
    strengths: list[str] = Field(description="잘한 점 1~2개. 각 항목은 '라벨: 설명' 형식이 아니라 완결된 서술형 문장으로 작성하되, 전체 항목을 합쳐 최소 2문장 이상이 되도록 할 것. 각 문장의 핵심 키워드는 **볼드**로 강조할 것")
    improvements: list[str] = Field(description="개선이 필요한 부분과 그 이유 1~2개. 각 항목은 '라벨: 설명' 형식이 아니라 완결된 서술형 문장으로 작성하되, 전체 항목을 합쳐 최소 2문장 이상이 되도록 할 것. 각 문장의 핵심 키워드는 **볼드**로 강조할 것")
    next_step: str = Field(description="다음 러닝을 위한 구체적인 목표나 조언 최소 2문장 이상. summary의 흐름을 이어받아 자연스럽게 마무리할 것. 핵심 키워드나 목표 수치는 **볼드**로 강조하고, 실천 항목이 2개 이상이면 문장으로 풀어쓰지 말고 '1. 2.' 번호 리스트로 작성할 것")


def _dtw_position_label(idx: int, length: int) -> str:
    ratio = idx / max(length - 1, 1)
    return "초반" if ratio < 1 / 3 else "중반" if ratio < 2 / 3 else "후반"


def _cadence_direction(v: float) -> str:
    # 가이드: 양수=한 걸음이 길어짐(감속), 음수=한 걸음이 짧아짐(가속)
    return "감속" if v > 0 else "가속" if v < 0 else "변화 없음"


def _amplitude_direction(v: float) -> str:
    # 가이드: 양수=평소보다 크게/높이 움직임, 음수=평소보다 덜 듦(끌림)
    return "평소보다 크게 움직임" if v > 0 else "평소보다 덜 듦 (끌림 가능성)" if v < 0 else "변화 없음"


def _shape_dev_note(shape: float | None) -> str | None:
    if shape is None or abs(shape) < _SHAPE_DEV_NOTABLE:
        return None
    return f"걸음 형태 변화 뚜렷함 (shapeDev={round(shape, 1)})"


def _cadence_dev_note(cadence: float | None) -> str | None:
    if cadence is None or abs(cadence) < _CADENCE_DEV_NOTABLE:
        return None
    return f"속도 {_cadence_direction(cadence)} 뚜렷함 (cadenceDev={round(cadence, 1):+})"


def _notable_devs(indices: list[int], ctxs: list[dict]) -> list[str]:
    """구간 평균 shapeDev/cadenceDev가 가이드의 임계값을 넘을 때만 '뚜렷하다'고 표기한다."""
    shape = mean(ctxs[i].get("shapeDev", 0) or 0 for i in indices)
    cadence = mean(ctxs[i].get("cadenceDev", 0) or 0 for i in indices)
    return [note for note in (_shape_dev_note(shape), _cadence_dev_note(cadence)) if note]


def _dtw_segments(times: list, ctxs: list[dict], k: int = _DTW_SEGMENTS) -> list[dict]:
    """윈도우 시계열을 k개 구간으로 나눠, 구간 내 '최악 상태'를 우선해 시간축 서사를 만든다.
    평균으로 뭉개면 alert 같은 위험 신호가 희석될 수 있어 worst-case로 집계한다."""
    n = len(ctxs)
    if n == 0:
        return []
    if n < k:
        groups = [[i] for i in range(n)]
    else:
        size = n / k
        groups = []
        for s in range(k):
            start = int(s * size)
            end = int((s + 1) * size) if s < k - 1 else n
            groups.append(list(range(start, end)))

    segments = []
    for group in groups:
        mid = group[len(group) // 2]
        avg_sim = round(mean(ctxs[i]["avgSimilarity"] for i in group), 1)
        worst_status = max((ctxs[i].get("status", "stable") for i in group), key=lambda s: _STATUS_SEVERITY.get(s, 0))
        amplitude = round(mean(ctxs[i].get("amplitudeDev", 0) or 0 for i in group), 1)
        segments.append({
            "position": _dtw_position_label(mid, n),
            "avg_similarity": avg_sim,
            "status": worst_status,
            "amplitude_dev": amplitude,
            "notable": _notable_devs(group, ctxs),
        })
    return segments


@traceable(name="summarize_dtw")
def summarize_dtw(windows: list[SessionWindow]) -> dict:
    seq = [(w.window_start_time, w.dtw_context) for w in windows if w.dtw_context and "avgSimilarity" in w.dtw_context]
    if not seq:
        return {"trend": "no_data"}

    times = [t for t, _ in seq]
    ctxs = [c for _, c in seq]
    sims = [c["avgSimilarity"] for c in ctxs]
    statuses = [c.get("status", "stable") for c in ctxs]
    trends = [c.get("trend", "steady") for c in ctxs]
    n = len(sims)

    third = max(n // 3, 1)
    start_avg, end_avg = round(mean(sims[:third]), 1), round(mean(sims[-third:]), 1)

    status_counts = Counter(statuses)
    status_ratio = {s: round(status_counts.get(s, 0) / n, 2) for s in ("stable", "drift", "alert")}
    trend_counts = Counter(trends)
    trend_ratio = {t: round(trend_counts.get(t, 0) / n, 2) for t in ("improving", "steady", "worsening")}

    # 가장 이탈이 컸던 지점: alert > drift > stable 우선, 그 안에서는 avgSimilarity가 가장 낮은 윈도우.
    # 임의의 수치 기준이 아니라 이미 모델이 판단한 status를 그대로 활용한다.
    worst_idx = min(range(n), key=lambda i: (-_STATUS_SEVERITY[statuses[i]], sims[i]))
    worst_ctx = ctxs[worst_idx]

    return {
        "start_avg": start_avg,
        "end_avg": end_avg,
        "session_trend": "increasing" if end_avg > start_avg else "decreasing" if end_avg < start_avg else "stable",
        "status_ratio": status_ratio,
        "trend_ratio": trend_ratio,
        "worst_moment": {
            "position": _dtw_position_label(worst_idx, n),
            "time": times[worst_idx],
            "status": worst_ctx.get("status"),
            "avg_similarity": worst_ctx.get("avgSimilarity"),
            "trend": worst_ctx.get("trend"),
            "shape_dev": worst_ctx.get("shapeDev"),
            "cadence_dev": worst_ctx.get("cadenceDev"),
            "amplitude_dev": worst_ctx.get("amplitudeDev"),
            "timeline": worst_ctx.get("timeline"),
        },
        "segments": _dtw_segments(times, ctxs),
    }


@traceable(name="resolve_docs")
def resolve_docs(tag_ratio: dict, kb: dict) -> tuple[list[dict], str, list[str]]:
    if tag_ratio.get("안정 러닝", 0) >= 0.7:
        return pick_docs("안정 러닝", kb), "encouragement", []

    # 태그 비율 상위 순으로 최대 2개를 골라 관련 문서를 선택한다.
    top_tags = sorted(
        ((t, r) for t, r in tag_ratio.items() if t and t != "안정 러닝"),
        key=lambda x: -x[1],
    )[:2]
    tags = [t for t, _ in top_tags]

    docs = [d for t in tags for d in pick_docs(t, kb)]
    if not docs:
        return pick_docs("안정 러닝", kb), "encouragement", []
    return docs, "corrective", tags


def _summarize_session(session: RunningSession) -> dict:
    distance_m = float(session.total_distance_m)
    duration_sec = session.total_duration_sec
    duration_min, remain_sec = divmod(duration_sec, 60)

    pace_display = "N/A"
    if distance_m > 0:
        pace_sec_per_km = round(duration_sec / (distance_m / 1000))
        pace_min, pace_sec = divmod(pace_sec_per_km, 60)
        pace_display = f"{pace_min}분 {pace_sec}초/km"

    return {
        "distance_km": round(distance_m / 1000, 2),
        "duration_display": f"{duration_min}분 {remain_sec}초",
        "pace_display": pace_display,
        "mode_label": "야외 러닝" if session.mode == "outdoor" else "실내 러닝",
    }


def _dominant_tag(tags: list[str]) -> str | None:
    """window의 대표 태그 하나를 고른다. '안정 러닝'은 다른 태그와 함께 나오지 않으므로 그대로 쓰고,
    문제 태그가 여러 개면 _ISSUE_TAG_PRIORITY 순서를 따른다."""
    if not tags:
        return None
    if _STABLE_TAG in tags:
        return _STABLE_TAG
    for t in _ISSUE_TAG_PRIORITY:
        if t in tags:
            return t
    return tags[0]


# 이 값(ms)보다 window 사이 간격이 벌어지면, 태그가 같아도 별도 구간으로 나눈다.
# 센서 끊김/앱 백그라운드 전환 등으로 한동안 window가 안 찍히다가 우연히 같은 태그로
# 재개되면, 시간 체크 없이 태그만 비교할 경우 그 공백 전체가 하나의 구간으로 잘못
# 합쳐져 타임라인에 비정상적으로 긴 구간이 나타난다.
_TIMELINE_MERGE_GAP_MS = 60_000


def _build_timeline(session: RunningSession, windows: list[SessionWindow]) -> list[dict]:
    """대표 태그가 이어지는 구간끼리 묶어 세션 시작 기준 분 단위 타임라인을 만든다."""
    if not windows:
        return []

    session_start_ms = kst_naive_to_epoch_ms(session.start_time)
    session_end_ms = kst_naive_to_epoch_ms(session.end_time)
    earliest_window_start_ms = min(w.window_start_time for w in windows)

    # 일부 클라이언트/기기 조합에서 session.startedAt보다 window timestamp가 먼저 들어오는
    # 경우가 있다. 타임라인은 사용자에게 보여주는 "러닝 시작 후 N분" 값이므로, 그런 경우
    # 가장 이른 window를 기준점으로 삼아 음수 구간이 저장되지 않게 한다.
    timeline_base_ms = min(session_start_ms, earliest_window_start_ms)
    max_elapsed_ms = max(session_end_ms, max(w.window_end_time for w in windows)) - timeline_base_ms

    segments: list[dict] = []
    for w in windows:
        tag = _dominant_tag(w.tags or [])
        if tag is None:
            continue
        if (
            segments
            and segments[-1]["tag"] == tag
            and w.window_start_time - segments[-1]["end_ms"] <= _TIMELINE_MERGE_GAP_MS
        ):
            segments[-1]["end_ms"] = w.window_end_time
        else:
            segments.append({"tag": tag, "start_ms": w.window_start_time, "end_ms": w.window_end_time})

    return [
        {
            "tag": seg["tag"],
            "start_min": round(max(0, seg["start_ms"] - timeline_base_ms) / 60000),
            "end_min": round(min(max_elapsed_ms, max(0, seg["end_ms"] - timeline_base_ms)) / 60000),
        }
        for seg in segments
    ]


def _to_feedback_out(fb: RunningFeedback, session_summary: dict, route: list[dict] | None = None) -> RunningFeedbackOut:
    """session_summary는 RunningSession에서 쿼리 없이 즉석 계산하는 값이라 저장하지 않고 매번 새로 채운다.
    timeline은 window 전체를 다시 읽어야 하는 비용이 있어 생성 시점에 RunningFeedback에 캐싱해두고 그대로 재사용한다."""
    return RunningFeedbackOut(
        feedback_id=fb.feedback_id,
        session_id=fb.session_id,
        user_id=fb.user_id,
        final_feedback=fb.final_feedback,
        tag_ratio=fb.tag_ratio,
        session_summary=session_summary,
        timeline=fb.timeline or [],
        route=route,
        created_at=fb.created_at,
    )


def get_session_feedback(db: Session, session: RunningSession) -> RunningFeedbackOut | None:
    """이미 생성된 피드백을 순수 조회만 한다 (LLM 생성/재생성 없음). 없으면 None."""
    fb = get_feedback_by_session(db, session.session_id)
    if not fb:
        return None
    session_summary = _summarize_session(session)
    return _to_feedback_out(fb, session_summary, session.route)


def _coaching_history(windows: list[SessionWindow], limit: int = _MAX_COACHING_HISTORY) -> list[dict]:
    history = []
    for w in windows:
        if w.feedback_text and w.tags:
            feedback = w.feedback_text.strip()
            if len(feedback) > _MAX_HISTORY_FEEDBACK_CHARS:
                feedback = feedback[:_MAX_HISTORY_FEEDBACK_CHARS].rstrip() + "…"
            history.append({
                "tags": w.tags,
                "feedback": feedback,
                "window_start": w.window_start_time,
            })
    if len(history) <= limit:
        return history
    # 토큰 절약을 위해 전 구간 커버리지를 유지하는 균등 샘플링 (처음 N개만 남기면 후반부 정보가 유실됨)
    step = (len(history) - 1) / (limit - 1) if limit > 1 else 0
    indices = sorted({round(i * step) for i in range(limit)})
    return [history[i] for i in indices]


def _build_user_profile(user) -> dict:
    today = date.today()
    age = today.year - user.birth_date.year - ((today.month, today.day) < (user.birth_date.month, user.birth_date.day))
    return {
        "age": age,
        "gender": "남성" if user.gender == "man" else "여성",
        "height": user.height,
        "weight": user.weight,
    }


def _build_dtw_prompt_lines(dtw_summary: dict) -> list[str]:
    if dtw_summary.get("trend") == "no_data":
        return ["", "[개인 기준선 대비 편차 분석] 데이터 없음"]

    status_ratio = dtw_summary.get("status_ratio", {})
    trend_ratio = dtw_summary.get("trend_ratio", {})
    worst = dtw_summary.get("worst_moment", {})

    lines = [
        "",
        "[개인 기준선 대비 편차 분석 (dtwContext)]",
        "- 이 지표는 절대적인 자세 판단이 아니라, 이 사용자의 평소 러닝 패턴(캘리브레이션된 개인 기준선) 대비 "
        "현재 구간이 얼마나 벗어났는지를 나타내는 상대적 편차입니다. 표현할 때도 '자세가 틀렸다'가 아니라 "
        "'평소보다 흐트러졌다'는 식으로 서술하세요.",
        "- avgSimilarity: 0~100 스케일, 100에 가까울수록 평소와 동일, 낮을수록 이탈이 큼.",
        f"- 세션 내 상태 비율(모델이 이미 판정한 값): 안정 {round(status_ratio.get('stable', 0) * 100)}%, "
        f"이탈 시작 {round(status_ratio.get('drift', 0) * 100)}%, 경고 {round(status_ratio.get('alert', 0) * 100)}%",
        f"- 추세 비율: 회복 중 {round(trend_ratio.get('improving', 0) * 100)}%, "
        f"유지 {round(trend_ratio.get('steady', 0) * 100)}%, 악화 중 {round(trend_ratio.get('worsening', 0) * 100)}%",
        f"- 전반부 평균 유사도 {dtw_summary.get('start_avg', 'N/A')} → 후반부 평균 유사도 {dtw_summary.get('end_avg', 'N/A')} "
        f"(세션 전체 흐름: {dtw_summary.get('session_trend', 'N/A')})",
    ]

    if worst:
        shape = worst.get("shape_dev")
        cadence = worst.get("cadence_dev")
        amplitude = worst.get("amplitude_dev")
        avg_similarity = worst.get("avg_similarity")
        lines.append(
            f"- 가장 이탈이 컸던 지점: {worst.get('position')} ({worst.get('time')}), "
            f"상태={_STATUS_LABELS.get(worst.get('status'), worst.get('status'))}, "
            f"유사도={round(avg_similarity, 1) if avg_similarity is not None else 'N/A'}, "
            f"해당 구간 추세={_TREND_LABELS.get(worst.get('trend'), worst.get('trend'))}"
        )
        dev_notes = [note for note in (_shape_dev_note(shape), _cadence_dev_note(cadence)) if note]
        if amplitude is not None:
            dev_notes.append(
                f"발 움직임: {_amplitude_direction(amplitude)} (amplitudeDev={round(amplitude, 1):+}, "
                "정해진 임계값 없음 — 단정하지 말고 상대적으로만 참고)"
            )
        lines += [f"  · {note}" for note in dev_notes]

        timeline = worst.get("timeline")
        if timeline:
            lines.append(
                f"  · 해당 20초 구간의 3초 간격 유사도 추이: {timeline} "
                "(완만한 하락=점진적 피로 누적, 급락=갑작스러운 폼 붕괴나 이상 상황으로 해석하세요)"
            )

    segments = dtw_summary.get("segments") or []
    if segments:
        lines.append("- 세션 흐름 (시간순, 구간 내 최악 상태 기준):")
        for seg in segments:
            notable = ", ".join(seg.get("notable", [])) or "특이사항 없음"
            lines.append(
                f"  · {seg['position']}: 유사도 {seg['avg_similarity']}, "
                f"상태 {_STATUS_LABELS.get(seg['status'], seg['status'])}, {notable}, "
                f"amplitudeDev {seg['amplitude_dev']:+} (임계값 미정, 참고용)"
            )
    return lines


def _build_feedback_prompt(
    session_id: int,
    session_summary: dict,
    tag_ratio: dict,
    dominant_tags: list[str],
    dtw_summary: dict,
    docs: list[dict],
    prompt_mode: str,
    coaching_history: list[dict] | None = None,
    user_profile: dict | None = None,
) -> str:
    lines = []
    if user_profile:
        lines += [
            "[사용자 프로필]",
            f"- 나이: {user_profile['age']}세",
            f"- 성별: {user_profile['gender']}",
            f"- 키: {user_profile['height']}cm",
            f"- 몸무게: {user_profile['weight']}kg",
            "",
        ]

    rounded_tag_ratio = {t: round(r, 2) for t, r in tag_ratio.items()}
    lines += [
        "[러닝 세션 요약]",
        f"- 세션 ID: {session_id}",
        f"- 거리: {session_summary['distance_km']}km",
        f"- 시간: {session_summary['duration_display']}",
        f"- 평균 페이스: {session_summary['pace_display']}",
        f"- 러닝 종류: {session_summary['mode_label']}",
        f"- 태그 비율: {json.dumps(rounded_tag_ratio, ensure_ascii=False)}",
        f"- 주요 태그: {', '.join(dominant_tags) if dominant_tags else '없음'}",
        f"- 코칭 모드: {prompt_mode}",
    ]

    lines += _build_dtw_prompt_lines(dtw_summary)

    lines += [
        "",
        "[참고 문헌]",
        *[f"- {d['text']}" for d in docs],
    ]

    if coaching_history:
        lines += [
            "",
            "[구간별 코칭 피드백 (이미 사용자에게 전달됨)]",
            *[
                f"- {h['window_start']} 지점 | 태그: {', '.join(h['tags'])} → {h['feedback']}"
                for h in coaching_history
            ],
            "",
            "※ 위 구간별 피드백은 러닝 중 이미 전달된 내용입니다. 같은 조언을 문장만 바꿔 반복하지 말고, "
            "위 dtwContext 분석(상태 비율, 가장 이탈이 컸던 지점, 구간별 흐름)과 연결지어 세션 전체를 관통하는 "
            "상위 수준의 패턴으로 재해석해서 제공하세요.",
        ]

    lines += [
        "",
        "위 정보를 바탕으로 아래 항목에 맞춰 러닝 세션 종합 피드백을 JSON으로 작성해주세요:",
        "- summary: 세션이 시간 순서대로 어떻게 흘러갔는지(초반 상태 → 중반 변화 → 그 결과)를 원인과 결과로 이어서 최소 2문장 이상으로 서술",
        "- strengths: 잘한 점 1~2개, 각각 완결된 문장으로, 항목을 합쳐 최소 2문장 이상",
        "- improvements: 개선점과 그 이유 1~2개, 각각 완결된 문장으로, 항목을 합쳐 최소 2문장 이상",
        "- next_step: summary의 흐름을 이어받아 다음 러닝에서 무엇을, 왜 해야 하는지 최소 2문장 이상으로 서술",
        "- 네 항목 모두 각각 최소 2문장 이상이어야 합니다. 한 문장으로 짧게 끝내지 마세요.",
        "",
        "summary/strengths/improvements는 전체적으로 이어 붙여 읽었을 때 하나의 자연스러운 글처럼 읽혀야 합니다. "
        "'착지 충격 완화:', '페이스 안정화:' 같은 항목명+콜론(라벨) 형식은 절대 쓰지 말고, "
        "'~했지만, ~하며, 그 영향으로 ~' 같은 접속 표현으로 문장과 문장을 자연스럽게 이어주세요.",
        "",
        "가독성을 위해 다음 서식을 지켜주세요:",
        "- 각 항목에서 사용자가 실제로 기억해야 할 핵심 키워드(구체적인 동작·부위)와 목표 수치(퍼센트, 횟수, 분/초 등)는 "
        "반드시 마크다운 굵게(**단어**)로 감싸세요. 문장 전체나 조사(을/를/이/가 등)까지 함께 감싸지 말고, "
        "항목당 많아야 2~3개만 강조하세요.",
        "- next_step에 다음 러닝에서 실천할 구체적 항목이 2개 이상이면, 문장으로 풀어쓰지 말고 "
        "'1. ...\\n2. ...' 형식의 번호 리스트로 작성하세요. 각 항목은 한 문장 이내로 짧게 쓰세요.",
    ]
    return "\n".join(lines)


_SYSTEM_INSTRUCTION = (
    "당신은 러닝 코칭 전문가입니다. 제공된 세션 태그 비율, 개인 기준선 대비 편차 분석(dtwContext), "
    "구간별 코칭 이력, 참고 문헌을 종합하여 사용자에게 과학적이고 구체적인 피드백을 한국어로 제공합니다.\n\n"
    "응답은 반드시 지정된 JSON 스키마(summary, strengths, improvements, next_step)를 따라 "
    "구조화해야 합니다. 단, 필드가 나뉘어 있다고 해서 각 필드를 독립된 항목 나열로 쓰지 마세요. "
    "네 필드를 순서대로 이어 붙였을 때 '초반엔 이랬는데, 그 결과 이렇게 됐고, 그래서 다음엔 이렇게 하면 좋겠다'는 "
    "하나의 자연스러운 이야기로 읽혀야 합니다. '착지 충격 완화:', '페이스 안정화:'처럼 항목명을 콜론(:)으로 "
    "붙이는 라벨링 형식은 쓰지 말고, 원인과 결과를 접속어로 연결한 완결된 문장으로 작성하세요.\n\n"
    "가독성을 위한 서식 규칙:\n"
    "- 각 필드에서 사용자가 실제로 기억하고 실천해야 할 핵심 키워드(구체적인 동작·부위)와 목표 수치(퍼센트, "
    "횟수, 분/초 등)는 반드시 마크다운 굵게 표시(**단어**)로 감싸세요. 문장 전체나 조사(을/를/이/가 등)까지 "
    "함께 감싸지 말고, 필드당 많아야 2~3개만 강조하세요.\n"
    "- next_step에서 다음 러닝을 위해 실천할 구체적 항목이 2개 이상이면, 문장으로 풀어쓰지 말고 순서가 있는 "
    "마크다운 리스트('1. ', '2. ' 를 각 줄 맨 앞에 붙이고 줄바꿈 \\n으로 구분)로 작성하세요. 리스트 항목 하나는 "
    "한 문장 이내로 짧게 쓰세요. summary/strengths/improvements는 리스트로 쪼개지 말고 위에서 요구한 "
    "서술형 문장 흐름을 그대로 유지하세요.\n\n"
    "dtwContext 관련 지표는 절대적인 자세 판단이 아니라 이 사용자 개인의 평소 패턴 대비 상대적 편차입니다. "
    "'자세가 틀렸다'가 아니라 '평소보다 흐트러졌다'는 식으로 표현하세요. status(stable/drift/alert)와 "
    "trend(improving/steady/worsening)는 이미 판정된 값이니 그대로 신뢰하고, 임의로 다른 기준을 만들지 마세요. "
    "여러 지표를 함께 보고 종합하세요. 예를 들어 cadenceDev 양수(감속) + amplitudeDev 음수(발을 덜 듦) + "
    "trend가 worsening이면 피로 누적 가능성으로 해석할 수 있습니다. shapeDev가 크면 걸음 형태 변화, "
    "cadenceDev가 크면 속도 변화, amplitudeDev가 크면 발 움직임 변화로 원인을 구분해서 짚어주세요. "
    "amplitudeDev에는 정해진 임계값이 없으니 그 값만으로 '뚜렷하다'고 단정하지 마세요.\n\n"
    "제공된 사용자 프로필(나이, 성별, 키, 몸무게)을 참고해 사용자의 체형과 체력 수준에 맞는 조언을 제공하세요. "
    "연령대에 맞는 적절한 강도를 제안하세요.\n\n"
    "prompt_mode가 'encouragement'이면 칭찬과 강점 유지 방법 위주로, "
    "'corrective'이면 개선점과 구체적인 교정 방법 위주로 작성하세요.\n\n"
    "'구간별 코칭 피드백'은 러닝 중 이미 사용자에게 전달된 조언입니다. 동일한 지시를 문장만 바꿔 반복하지 말고, "
    "함께 제공되는 dtwContext 분석과 연결지어 세션 전체를 관통하는 상위 수준의 인사이트로 재구성하세요."
)


def _is_retryable_error(e: Exception) -> bool:
    """일시적 오류(레이트리밋 429, 서버 오류 5xx, 타임아웃/연결 오류)만 재시도 대상으로 삼는다.
    인증 오류(401/403)나 잘못된 요청(400) 등은 재시도해도 같은 결과가 나오므로 즉시 포기해서
    불필요한 대기 시간을 만들지 않는다."""
    status = getattr(e, "code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES
    # SDK 예외에서 상태 코드를 못 얻는 경우, 타임아웃/연결류 예외 이름으로만 보수적으로 판별한다.
    name = type(e).__name__.lower()
    return any(keyword in name for keyword in ("timeout", "connection", "deadline"))


@traceable(name="call_llm")
def _call_llm(prompt: str) -> _FeedbackOutput | None:
    last_error: Exception | None = None
    # 재시도까지 포함한 전체 소요시간이 프론트 타임아웃을 넘지 않도록 하는 전체 예산 마감시각.
    # 시도당 타임아웃(_LLM_TIMEOUT_MS)만으로는 재시도가 겹칠 때 총 소요시간이 계속 늘어나는 걸
    # 막을 수 없어서, 이 데드라인으로 "몇 번을 재시도하든 총합은 이 안에서 끝난다"를 보장한다.
    deadline = time.monotonic() + _LLM_TOTAL_BUDGET_SEC

    for attempt in range(_LLM_MAX_RETRIES + 1):
        remaining_sec = deadline - time.monotonic()
        if remaining_sec <= 1:
            # 남은 예산이 거의 없으면 시도해봐야 곧바로 잘릴 뿐이므로 바로 포기한다.
            logger.warning(
                "LLM 호출 예산 소진으로 재시도 중단 (세션 피드백, 시도 %d/%d 진입 전, 남은 예산=%.1fs)",
                attempt + 1, _LLM_MAX_RETRIES + 1, remaining_sec,
            )
            break
        # 이번 시도에는 "시도당 상한"과 "남은 전체 예산" 중 더 짧은 쪽을 타임아웃으로 준다.
        attempt_timeout_ms = int(min(_LLM_TIMEOUT_MS, remaining_sec * 1000))
        try:
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    temperature=0.3,
                    max_output_tokens=4000,
                    response_mime_type="application/json",
                    response_schema=_FeedbackOutput,
                    http_options=genai.types.HttpOptions(timeout=attempt_timeout_ms),
                ),
            )
            parsed = getattr(response, "parsed", None)
            if parsed is None and response.text:
                parsed = _FeedbackOutput.model_validate_json(response.text)
            return parsed
        except Exception as e:
            last_error = e
            retryable = _is_retryable_error(e)
            logger.warning(
                "LLM 호출 실패 (세션 피드백, 시도 %d/%d, retryable=%s): %s",
                attempt + 1, _LLM_MAX_RETRIES + 1, retryable, e,
            )
            if not retryable or attempt == _LLM_MAX_RETRIES:
                break
            # 지수 백오프 + 지터: 동시 실패한 여러 요청이 한꺼번에 재시도로 몰리는 걸 방지
            sleep_sec = _LLM_RETRY_BACKOFF_BASE_SEC * (2 ** attempt) + random.uniform(0, 0.5)
            remaining_sec = deadline - time.monotonic()
            if sleep_sec >= remaining_sec:
                # 백오프만으로 남은 예산을 다 써버리는 경우 재시도할 시간이 없으므로 바로 포기한다.
                break
            time.sleep(sleep_sec)

    logger.error("LLM 호출 최종 실패 (세션 피드백): %s", last_error)
    return None


@traceable(name="generate_session_feedback", project_name="azas-feedback")
def generate_session_feedback(
    db: Session, session: RunningSession, user_id, user=None, force_regenerate: bool = False
) -> RunningFeedbackOut:
    # 호출부(API 레이어)에서 이미 세션을 조회/검증했으므로 여기서 다시 조회하지 않는다.
    session_id = session.session_id
    # session_summary는 RunningSession에서 쿼리 없이 즉석 계산하는 값이라 캐시 히트 여부와 무관하게
    # 매 반환 지점에서 최신값으로 채운다 (timeline은 비용이 있어 RunningFeedback에 캐싱, _to_feedback_out 참고).
    session_summary = _summarize_session(session)

    # 이미 생성된 피드백을 다시 읽어오는 흔한 경로(force_regenerate=False)는 락을 잡을 필요가 없다.
    # 락과 그 락이 걸려 있는 트랜잭션은 LLM 호출이 끝날 때까지 DB 커넥션을 붙잡고 있으므로,
    # 캐시 조회만 하는 요청까지 그 비용을 치르지 않도록 락 밖에서 먼저 확인한다.
    if not force_regenerate:
        existing = get_feedback_by_session(db, session_id)
        if existing:
            return _to_feedback_out(existing, session_summary, session.route)

    # 동일 세션에 대한 동시 요청이 각자 LLM을 중복 호출하지 않도록 트랜잭션 범위 락으로 직렬화한다.
    # 커밋/롤백 시 자동 해제되므로 별도 unlock이 필요 없다.
    db.execute(sa_text("SELECT pg_advisory_xact_lock(:sid)"), {"sid": session_id})

    if not force_regenerate:
        # 락 대기 중 다른 요청이 먼저 생성을 마쳤을 수 있으므로 다시 확인한다 (double-checked locking).
        existing = get_feedback_by_session(db, session_id)
        if existing:
            return _to_feedback_out(existing, session_summary, session.route)

    windows = get_windows_by_session(db, session_id)
    if not windows:
        no_data = {
            "summary": INSUFFICIENT_DATA_TEXT,
            "strengths": [],
            "improvements": [],
            "next_step": "",
        }
        data = {"final_feedback": json.dumps(no_data, ensure_ascii=False), "tag_ratio": {}, "timeline": []}
        fb = upsert_feedback(db, session_id, user_id, data)
        return _to_feedback_out(fb, session_summary, session.route)

    all_tags = [t for w in windows for t in (w.tags or [])]
    total = len(all_tags) or 1
    tag_ratio = {t: c / total for t, c in Counter(all_tags).items()}

    dtw_summary = summarize_dtw(windows)
    coaching_history = _coaching_history(windows)
    kb = get_kb()
    docs, prompt_mode, effective_tags = resolve_docs(tag_ratio, kb)
    user_profile = _build_user_profile(user) if user is not None else None
    prompt = _build_feedback_prompt(
        session_id, session_summary, tag_ratio, effective_tags, dtw_summary, docs, prompt_mode, coaching_history,
        user_profile=user_profile,
    )

    timeline = _build_timeline(session, windows)

    llm_result = _call_llm(prompt)
    if llm_result is None:
        fallback_data = {
            "summary": FALLBACK_TEXT,
            "strengths": [],
            "improvements": [],
            "next_step": "다음 러닝 후 다시 시도해주세요.",
        }
        data = {"final_feedback": json.dumps(fallback_data, ensure_ascii=False), "tag_ratio": tag_ratio, "timeline": timeline}
        fb = upsert_feedback(db, session_id, user_id, data)
        return _to_feedback_out(fb, session_summary, session.route)

    feedback_dict = llm_result.model_dump()

    data = {
        "final_feedback": json.dumps(feedback_dict, ensure_ascii=False),
        "tag_ratio": tag_ratio,
        "timeline": timeline,
    }
    fb = upsert_feedback(db, session_id, user_id, data)
    return _to_feedback_out(fb, session_summary, session.route)
