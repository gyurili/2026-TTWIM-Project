import logging

import numpy as np
from google import genai

from app.core.config import settings

logger = logging.getLogger(__name__)

COACHING_TYPES: dict[str, str] = {
    "form": "자세",
    "cadence": "보폭",
    "impact": "충격",
    "breathing": "호흡",
    "encourage": "격려",
}




def extract_stats(window: list[list[float]]) -> dict:
    arr = np.array(window, dtype=float)
    acc_z = arr[:, 2]
    gyro = arr[:, 3:6]

    acc_z_mean = float(np.mean(acc_z))
    acc_z_max = float(np.max(acc_z))

    gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))
    gyro_mag_peak = float(np.max(gyro_mag))

    acc_mag = np.sqrt(np.sum(arr[:, :3] ** 2, axis=1))
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


def _build_prompt(stats: dict, ratio: float) -> str:
    return (
        f"[센서 통계]\n"
        f"- 수직 충격(acc_z): 평균={stats['acc_z_mean']}g, 최대={stats['acc_z_max']}g\n"
        f"- 회전 충격(gyro): 최대 {stats['gyro_mag_peak']}deg/s\n"
        f"- 충격 지수: {stats['impact_score']}\n"
        f"- 걸음 수: {stats['step_count']}보\n"
        f"- 보폭 규칙성: {stats['stride_regularity']}\n"
        f"- 비정상 비율: {ratio:.0%}\n\n"
        f"위 수치를 분석하여 사용자의 러닝 자세를 과학적으로 교정해주세요."
    )


def _classify_coaching_type(stats: dict) -> str:
    if stats.get("impact_score", 0) > 10 or stats.get("acc_z_max", 0) > 8:
        return "impact"
    if stats.get("stride_regularity", 1) < 0.3:
        return "form"
    if stats.get("step_count", 0) > 60:
        return "cadence"
    return "encourage"


def _call_llm(prompt: str, coaching_type: str) -> tuple[str, str] | None:
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=(
                    f"당신은 러닝 코칭 전문가입니다. 센서 데이터를 분석하여 "
                    f"사용자에게 과학적이고 구체적인 교정 동작을 한국어로 알려줍니다. "
                    f"센서 수치를 직접 언급하며, 사용자가 즉시 실행할 수 있는 동작 위주로 "
                    f"2문장 이내로 간결하게 조언하세요. "
                    f"오늘의 코칭 주제는 '{COACHING_TYPES.get(coaching_type, coaching_type)}'입니다."
                ),
                temperature=0.3,
                max_output_tokens=150,
            ),
        )
        text = response.text
        if not text:
            return None
        return text.strip(), coaching_type
    except Exception as e:
        logger.error("LLM 호출 실패: %s", e)
        return None


def generate_coaching(stats: dict, ratio: float) -> tuple[str, str] | None:
    coaching_type = _classify_coaching_type(stats)
    prompt = _build_prompt(stats, ratio)
    return _call_llm(prompt, coaching_type)
