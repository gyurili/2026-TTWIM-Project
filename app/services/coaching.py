import json
import logging

import numpy as np
from google import genai
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

COACHING_TAGS: list[str] = ["안정 러닝", "충격 과다 러닝", "페이스 변화 러닝"]

_DTW_STATUS_DESC = {
    "stable": "평소와 거의 같음(정상)",
    "drift": "평소에서 벗어나기 시작함(주의 관찰 필요)",
    "alert": "평소와 크게 다름(개입 권장)",
}
_DTW_TREND_DESC = {
    "improving": "유사도 상승 중(폼 회복 추세)",
    "steady": "큰 변화 없음",
    "worsening": "유사도 하락 중(폼 붕괴 추세)",
}


class _CoachingOutput(BaseModel):
    text: str
    tags: list[str]


def extract_stats(channels: dict[str, list[float]]) -> dict:
    acc_z = np.array(channels["acc_z"], dtype=float)
    gyro = np.array([channels["gyro_x"], channels["gyro_y"], channels["gyro_z"]], dtype=float).T
    acc = np.array([channels["acc_x"], channels["acc_y"], channels["acc_z"]], dtype=float).T

    acc_z_mean = float(np.mean(acc_z))
    acc_z_max = float(np.max(acc_z))

    gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))
    gyro_mag_peak = float(np.max(gyro_mag))

    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))
    impact_score = float(np.percentile(acc_mag, 95))

    zero_crossings = np.sum(np.diff(np.sign(acc_z)) != 0)
    step_count = int(zero_crossings)

    autocorr = np.correlate(acc_z, acc_z, mode="full")
    autocorr = autocorr[len(acc_z) - 1:]
    autocorr = autocorr / (autocorr[0] + 1e-8)
    if len(autocorr) > 1:
        stride_regularity = max(0.0, min(1.0, float(autocorr[1])))
        if np.isnan(stride_regularity):
            stride_regularity = 0.0
    else:
        stride_regularity = 0.0

    return {
        "acc_z_mean": round(acc_z_mean, 4),
        "acc_z_max": round(acc_z_max, 4),
        "gyro_mag_peak": round(gyro_mag_peak, 4),
        "impact_score": round(impact_score, 4),
        "step_count": step_count,
        "stride_regularity": round(stride_regularity, 4),
    }


def _build_prompt(stats: dict, ratio: float, dtw: dict) -> str:
    lines = [
        "[센서 통계 (ML 모델이 비정상으로 판정한 구간)]",
        f"- 수직 충격(acc_z): 평균={stats['acc_z_mean']}g, 최대={stats['acc_z_max']}g",
        f"- 회전 충격(gyro): 최대 {stats['gyro_mag_peak']}deg/s",
        f"- 충격 지수: {stats['impact_score']}",
        f"- 걸음 수: {stats['step_count']}보",
        f"- 보폭 규칙성: {stats['stride_regularity']}",
        f"- 비정상 비율: {ratio:.0%}",
        "",
        "[개인 기준선 대비 편차 (DTW, 이 사용자의 평소 패턴 대비 상대적 이탈)]",
        f"- 상태: {dtw['status']} ({_DTW_STATUS_DESC.get(dtw['status'], dtw['status'])})",
        f"- 평균 유사도: {dtw['avgSimilarity']} (0~100, 높을수록 평소와 동일)",
        f"- 추세: {dtw['trend']} ({_DTW_TREND_DESC.get(dtw['trend'], dtw['trend'])})",
        f"- 걸음 형태 편차(shapeDev): {dtw['shapeDev']} (0에 가까울수록 평소와 동일, 대략 2 이상이면 뚜렷한 형태 변화)",
        f"- 속도 편차(cadenceDev): {dtw['cadenceDev']} (양수=감속/걸음이 길어짐, 음수=가속/걸음이 짧아짐, 대략 ±3 이상이면 뚜렷한 변화)",
        f"- 발 움직임 편차(amplitudeDev): {dtw['amplitudeDev']} (양수=평소보다 크게 움직임, 음수=평소보다 덜 듦)",
        f"- 유사도 타임라인(3초 간격): {dtw['timeline']} (완만한 하락=점진적 피로, 급락=갑작스러운 폼 붕괴/이상 상황)",
        "",
        "위 수치를 종합적으로 분석하여 사용자의 러닝 자세를 과학적으로 교정해주세요.",
    ]
    return "\n".join(lines)


def _call_llm(prompt: str) -> tuple[str, list[str]] | None:
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=(
                    "당신은 러닝 코칭 전문가입니다. 센서 통계와 개인 기준선 대비 편차(DTW) 정보를 "
                    "종합해 사용자에게 과학적이고 구체적인 교정 동작을 한국어로 알려줍니다. "
                    "DTW 편차 정보는 절대적 자세 판단이 아니라 '이 사용자가 평소보다 얼마나 흐트러졌는가'를 "
                    "뜻하므로, '자세가 틀렸다'가 아니라 '평소보다 흐트러졌다' 식으로 표현하세요. "
                    "cadenceDev 부호(+는 감속, -는 가속)와 amplitudeDev 부호(+는 크게 움직임, -는 덜 움직임)에 "
                    "유의하고, trend와 timeline으로 완만한 저하인지 급격한 붕괴인지 구분하세요. "
                    "예를 들어 cadenceDev 양수 + amplitudeDev 음수 + trend worsening이면 "
                    "속도가 느려지며 발을 덜 드는 중이므로 피로 누적 가능성으로 해석할 수 있습니다. "
                    "사용자가 즉시 실행할 수 있는 동작 위주로 2문장 이내로 간결하게 조언하세요. "
                    "말 끝을 흐리지 말고 반드시 마침표('.')로 완전히 끝맺음해야 합니다. "
                    "그리고 분석 결과를 바탕으로 tags를 선택하세요. 태그별 정의는 다음과 같습니다: "
                    "'충격 과다 러닝'은 충격 지수나 acc_z 최대값이 크거나 amplitudeDev가 뚜렷할 때, "
                    "'페이스 변화 러닝'은 cadenceDev나 shapeDev가 뚜렷하거나 trend가 worsening/improving일 때, "
                    "'안정 러닝'은 이 두 문제 중 어느 것도 뚜렷하지 않고 status가 stable에 가까울 때만 선택합니다. "
                    "'안정 러닝'은 문제가 없다는 뜻이므로 '충격 과다 러닝'이나 '페이스 변화 러닝'과 "
                    "동시에 선택할 수 없습니다. 문제가 하나라도 있으면 '안정 러닝'은 절대 포함하지 마세요. "
                    "해당하는 문제 태그가 여러 개면 모두 담아도 되지만, 최소 1개의 태그는 반드시 있어야 합니다."
                ),
                temperature=0.3,
                max_output_tokens=4000,
                response_mime_type="application/json",
                response_schema=_CoachingOutput,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None and response.text:
            parsed = _CoachingOutput(**json.loads(response.text))
        if parsed is None or not parsed.text:
            return None

        tags = [t for t in parsed.tags if t in COACHING_TAGS]
        return parsed.text.strip(), tags
    except Exception as e:
        logger.error("LLM 호출 실패: %s", e)
        return None


def generate_coaching(channels: dict[str, list[float]], ratio: float, dtw: dict) -> tuple[str, list[str]] | None:
    stats = extract_stats(channels)
    prompt = _build_prompt(stats, ratio, dtw)
    return _call_llm(prompt)