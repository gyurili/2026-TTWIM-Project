import json
import logging

import numpy as np
from scipy.signal import find_peaks
from google import genai
from openai import OpenAI
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

COACHING_TAGS: list[str] = ["안정 러닝", "충격 과다 러닝", "발끌기 러닝", "좁은 보폭 러닝", "페이스 변화 러닝"]

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

# --- 수치 안정성 관련 상수 ---
_EPS = 1e-8
_DEFAULT_SAMPLE_RATE_HZ = 50.0          # channels 에 샘플레이트 정보가 없어 임시로 가정하는 값
_MIN_SECONDS_REQUIRED = 1.5             # 최소 이 정도 길이(초)는 있어야 걸음 관련 지표를 신뢰할 수 있음
_MIN_STEPS_REQUIRED = 2                 # 이 미만이면 "달리고 있었다"고 보기 어려움
_LOW_MOTION_ACC_Z_STD = 0.05            # 이 이하이면 정지/거치 상태로 간주(g 단위 가정)
_MIN_GYRO_FLOOR = 1e-3                  # swing/stance 비율 분모 폭주 방지용 하한
_MIN_STEP_DISTANCE_SECONDS = 0.25       # 240spm보다 빠른 피크는 한 걸음으로 보지 않음
_STEP_PROMINENCE_RATIO = 0.12           # 수직 가속 범위 대비 최소 착지 피크 두드러짐
_MIN_STEP_PROMINENCE = 0.15             # 너무 작은 흔들림을 걸음 피크로 세지 않기 위한 하한(g)
_CHOPPY_BASE_CADENCE_SPM = 160.0        # 이 케이던스 이하는 좁은 보폭 지표를 0으로 둠


class InsufficientDataError(ValueError):
    """센서 구간이 너무 짧거나(샘플 수 부족) 실질적인 움직임이 없어(정지 상태) 지표를 신뢰할 수 없을 때 발생."""


class _CoachingOutput(BaseModel):
    text: str
    tags: list[str]


def _safe_div(numerator: float, denominator: float, floor: float = _EPS) -> float:
    """분모가 0에 가까워 값이 폭주하는 것을 막는 나눗셈. 분모의 절대값을 floor 이상으로 clip."""
    denom = denominator if abs(denominator) >= floor else (floor if denominator >= 0 else -floor)
    return float(numerator / denom)


def _finite_round(x: float, nd: int = 4) -> float:
    """NaN/Inf 가 프롬프트나 LLM 응답으로 새어나가지 않도록 하는 마지막 방어선."""
    x = float(x)
    if not np.isfinite(x):
        return 0.0
    return round(x, nd)


def extract_stats(
    channels: dict[str, list[float]],
    sample_rate_hz: float = _DEFAULT_SAMPLE_RATE_HZ,
) -> dict:
    """
    센서 채널로부터 러닝 폼 분석용 피처를 계산한다.

    Raises:
        InsufficientDataError: 구간이 너무 짧거나(최소 길이 미만) 실질적인 움직임이
            감지되지 않을 때(정지/거치 상태 등). 이 경우 호출측(generate_coaching)에서
            LLM을 호출하지 않고 별도로 처리해야 한다 — 잘못된 통계로 LLM이 태그를
            오판(예: 정지 상태를 '좁은 보폭 러닝'으로 오인)하는 것을 코드 레벨에서 차단.
    """
    required = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    missing = [k for k in required if k not in channels]
    if missing:
        raise InsufficientDataError(f"필수 채널 누락: {missing}")

    acc_x = np.asarray(channels["acc_x"], dtype=float)
    acc_y = np.asarray(channels["acc_y"], dtype=float)
    acc_z = np.asarray(channels["acc_z"], dtype=float)
    gyro_x = np.asarray(channels["gyro_x"], dtype=float)
    gyro_y = np.asarray(channels["gyro_y"], dtype=float)
    gyro_z = np.asarray(channels["gyro_z"], dtype=float)

    lengths = {len(acc_x), len(acc_y), len(acc_z), len(gyro_x), len(gyro_y), len(gyro_z)}
    if len(lengths) != 1:
        raise InsufficientDataError(f"채널 간 길이가 서로 다름: {lengths}")

    total_frames = len(acc_z)
    min_frames_required = max(10, int(sample_rate_hz * _MIN_SECONDS_REQUIRED))
    if total_frames < min_frames_required:
        raise InsufficientDataError(
            f"프레임 수 부족: {total_frames} < {min_frames_required} (최소 {_MIN_SECONDS_REQUIRED}초 필요)"
        )

    # NaN/Inf 가 입력 자체에 섞여 들어오는 경우 방어
    all_channels = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1)
    if not np.all(np.isfinite(all_channels)):
        raise InsufficientDataError("입력 채널에 NaN/Inf 값이 포함되어 있음")

    gyro = np.stack([gyro_x, gyro_y, gyro_z], axis=1)
    acc = np.stack([acc_x, acc_y, acc_z], axis=1)

    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))

    # --- 1) 충격량 및 일관성 분석 ---
    acc_z_mean = float(np.mean(acc_z))
    acc_z_max = float(np.max(acc_z))
    acc_z_min = float(np.min(acc_z))
    # min/max 대신 p95-p5 사용: 센서 노이즈로 인한 단발성 스파이크 한 프레임이
    # range를 왜곡해 choppy_step_index 등 파생 지표를 오염시키는 것을 방지(이상치에 강건).
    acc_z_range_robust = float(np.percentile(acc_z, 95) - np.percentile(acc_z, 5))
    acc_z_std = float(np.std(acc_z))
    acc_mag_rms = float(np.sqrt(np.mean(acc_mag ** 2)))
    impact_score = float(np.percentile(acc_mag, 95))

    jerk = np.abs(np.diff(acc_mag))
    jerk_p95 = float(np.percentile(jerk, 95)) if len(jerk) > 0 else 0.0

    # --- 2) 다리 스윙 및 지면 마찰 분석 (발끌기 정밀 지표) ---
    gyro_mag_mean = float(np.mean(gyro_mag))

    swing_mask = acc_mag < np.median(acc_mag)
    stance_mask = ~swing_mask
    swing_gyro_energy = float(np.mean(gyro_mag[swing_mask])) if np.any(swing_mask) else gyro_mag_mean
    stance_gyro_energy = float(np.mean(gyro_mag[stance_mask])) if np.any(stance_mask) else gyro_mag_mean
    swing_stance_gyro_ratio = _safe_div(swing_gyro_energy, stance_gyro_energy, floor=_MIN_GYRO_FLOOR)
    jerk_x = np.diff(acc_x)
    jerk_y = np.diff(acc_y)
    horizontal_jerk_std = float(np.std(np.sqrt(jerk_x ** 2 + jerk_y ** 2))) if len(jerk_x) > 0 else 0.0
    acc_z_low_p10 = float(np.percentile(acc_z, 10))
    acc_z_centered = acc_z - acc_z_mean
    duration_sec = total_frames / sample_rate_hz

    # --- 3) 보폭 및 케이던스 리듬 분석 (좁은 보폭/종종걸음 판별 지표) ---
    min_peak_distance = max(1, int(sample_rate_hz * _MIN_STEP_DISTANCE_SECONDS))
    peak_prominence = max(_MIN_STEP_PROMINENCE, acc_z_range_robust * _STEP_PROMINENCE_RATIO)
    step_peaks, _ = find_peaks(
        acc_z_centered,
        distance=min_peak_distance,
        prominence=peak_prominence,
    )
    step_count = int(len(step_peaks))
    cadence_spm = float(step_count / duration_sec * 60.0) if duration_sec > 0 else 0.0

    if len(step_peaks) > 2:
        intervals = np.diff(step_peaks)
        peak_interval_cv = float(np.std(intervals) / (np.mean(intervals) + _EPS))
    else:
        peak_interval_cv = 0.0

    cadence_excess = max(0.0, cadence_spm - _CHOPPY_BASE_CADENCE_SPM)
    rhythm_regularity_factor = _safe_div(1.0, 1.0 + peak_interval_cv)
    choppy_step_index = cadence_excess * rhythm_regularity_factor

    autocorr = np.correlate(acc_z_centered, acc_z_centered, mode="full")
    autocorr = autocorr[len(acc_z_centered) - 1:]
    stride_regularity = 0.0
    if len(autocorr) > 1 and autocorr[0] > 0:
        min_lag = max(1, int(0.2 * sample_rate_hz))
        max_lag = min(len(autocorr) - 1, int(2.0 * sample_rate_hz))
        if max_lag > min_lag:
            window = autocorr[min_lag:max_lag]
            best_lag_value = float(np.max(window))
            stride_regularity = max(0.0, min(1.0, best_lag_value / autocorr[0]))

    # --- 4) 페이스 및 시간적 변화 분석 ---
    acc_mag_mean = float(np.mean(acc_mag))
    acc_mag_cv = _safe_div(float(np.std(acc_mag)), acc_mag_mean, floor=_EPS)
    N = len(acc_mag)
    if N > 10:
        back_energy = float(np.mean(acc_mag[N // 2:] ** 2))
        front_back_ratio = _safe_div(float(np.mean(acc_mag[:N // 2] ** 2)), back_energy, floor=_EPS)
    else:
        front_back_ratio = 1.0

    low_motion = acc_z_std < _LOW_MOTION_ACC_Z_STD or step_count < _MIN_STEPS_REQUIRED
    if low_motion:
        raise InsufficientDataError(
            f"실질적인 러닝 동작이 감지되지 않음 (acc_z_std={acc_z_std:.4f}, step_count={step_count})"
        )

    # [정규화 지표] 절대 크기(g, deg/s) 기반 지표는 체중/키/보폭 크기에 따라
    # 사람마다 기준선 자체가 달라 절대 임계값을 두면 일반화가 안 됨.
    # 같은 구간의 전체 충격 에너지(acc_mag_rms)로 나눠 "이 사람 기준 상대적으로
    # 얼마나 튀는가"로 바꾸면 체형과 무관하게 비교 가능해짐.
    jerk_p95_norm = _safe_div(jerk_p95, acc_mag_rms)
    acc_z_std_norm = _safe_div(acc_z_std, acc_mag_rms)
    horizontal_jerk_std_norm = _safe_div(horizontal_jerk_std, acc_mag_rms)

    return {
        "acc_z_mean": _finite_round(acc_z_mean),
        "acc_z_max": _finite_round(acc_z_max),
        "acc_z_min": _finite_round(acc_z_min),
        "acc_z_low_p10": _finite_round(acc_z_low_p10),
        "acc_z_range": _finite_round(acc_z_range_robust),
        "acc_z_std": _finite_round(acc_z_std),
        "acc_z_std_norm": _finite_round(acc_z_std_norm),
        "impact_score": _finite_round(impact_score),
        "jerk_p95": _finite_round(jerk_p95),
        "jerk_p95_norm": _finite_round(jerk_p95_norm),
        "horizontal_jerk_std": _finite_round(horizontal_jerk_std),
        "horizontal_jerk_std_norm": _finite_round(horizontal_jerk_std_norm),
        "acc_mag_rms": _finite_round(acc_mag_rms),
        "gyro_mag_mean": _finite_round(gyro_mag_mean),
        "swing_gyro_energy": _finite_round(swing_gyro_energy),
        "swing_stance_gyro_ratio": _finite_round(swing_stance_gyro_ratio),
        "step_count": step_count,
        "cadence_spm": _finite_round(cadence_spm, 1),
        "choppy_step_index": _finite_round(choppy_step_index),
        "stride_regularity": _finite_round(stride_regularity),
        "peak_interval_cv": _finite_round(peak_interval_cv),
        "acc_mag_cv": _finite_round(acc_mag_cv),
        "front_back_ratio": _finite_round(front_back_ratio),
    }


def _build_prompt(stats: dict, ratio: float, dtw: dict) -> str:
    lines = [
        "[센서 통계: ML 비정상 판정 구간의 물리적 특징]",
        "1. 충격량 및 일관성 (충격 과다 판별 지표)",
        f" - 95% 최대 충격(impact_score): {stats['impact_score']}g",
        f" - 지속성 관절 충격량(jerk_p95): {stats['jerk_p95']} / 정규화값(jerk_p95_norm, ÷acc_mag_rms): {stats['jerk_p95_norm']} (체중·키에 따라 절대값 기준선이 달라지므로 반드시 정규화값 기준으로 판단. 클수록 지속적인 충격 과다)",
        f" - 수직 가속 범위(acc_z_range, p5~p95 기준): {stats['acc_z_range']}g (최대={stats['acc_z_max']}g, 최소={stats['acc_z_min']}g / 발을 안 들수록 이 범위가 극도로 좁아짐)",
        f" - 수직 충격 변동성(acc_z_std): {stats['acc_z_std']} / 정규화값(acc_z_std_norm, ÷acc_mag_rms): {stats['acc_z_std_norm']}",
        f" - 전체 충격 에너지(acc_mag_rms): {stats['acc_mag_rms']} (사람마다 체중/보폭에 따라 절대 크기가 다르므로 이 값 자체를 '이 사람의 기준 충격량'으로 삼아 다른 지표를 정규화함)",
        "",
        "2. 다리 스윙 및 리듬 (발끌기 vs 좁은 보폭 판단 핵심 지표)",
        f" - 다리 스윙 평균 속도(gyro_mag_mean): {stats['gyro_mag_mean']} deg/s",
        f" - 스윙 구간 자이로 에너지(swing_gyro_energy): {stats['swing_gyro_energy']} deg/s (단독으로 발끌기를 의미하지 않으며 케이던스/충격 패턴과 함께 해석)",
        f" - 스윙/스탠스 자이로 비율(swing_stance_gyro_ratio): {stats['swing_stance_gyro_ratio']} (이미 비율이라 체형과 무관하게 비교 가능. 낮으면 스윙 약화 후보이나, 충격 과다 착지나 짧은 보폭에서도 낮게 보일 수 있음)",
        f" - 수평 마찰 진동(horizontal_jerk_std): {stats['horizontal_jerk_std']} / 정규화값(horizontal_jerk_std_norm, ÷acc_mag_rms): {stats['horizontal_jerk_std_norm']} (수평 방향 흔들림. 발끌기뿐 아니라 강한 착지 충격에서도 커질 수 있으므로 단독 근거로 사용 금지)",
        f" - 수직 가속 하위 10%(acc_z_low_p10): {stats['acc_z_low_p10']}g",
        f" - 걸음 수: {stats['step_count']}보",
        f" - 케이던스(cadence_spm): {stats['cadence_spm']} spm (일반적인 러닝 케이던스는 약 160~180spm. 이보다 뚜렷하게 높을 때만 '빠르게 발을 구른다'고 판단)",
        f" - 종종걸음 지수(choppy_step_index): {stats['choppy_step_index']} (★핵심: 케이던스가 높고 보폭 간격이 비교적 일정할수록 증가합니다. 15 이상이면 좁은 보폭/종종걸음의 강한 근거입니다)",
        f" - 보폭 간격 변동계수(peak_interval_cv): {stats['peak_interval_cv']}",
        f" - 보폭 규칙성(stride_regularity): {stats['stride_regularity']}",
        "",
        "3. 페이스 및 시간적 변화",
        f" - 에너지 변동계수(acc_mag_cv): {stats['acc_mag_cv']}",
        f" - 전/후반부 에너지 비율(front_back_ratio): {stats['front_back_ratio']}",
        f" - 구간 내 비정상 비율: {ratio:.0%}",
        "",
        "[개인 기준선 대비 편차 (DTW)]",
        f"- 상태: {dtw['status']} ({_DTW_STATUS_DESC.get(dtw['status'], dtw['status'])})",
        f"- 평균 유사도: {dtw['avgSimilarity']}",
        f"- 추세: {dtw['trend']} ({_DTW_TREND_DESC.get(dtw['trend'], dtw['trend'])})",
        f"- 걸음 형태 편차(shapeDev): {dtw['shapeDev']}",
        f"- 속도 편차(cadenceDev): {dtw['cadenceDev']} (음수일수록 평소보다 발을 바쁘게 구르는 종종걸음 패턴)",
        f"- 발 움직임 편차(amplitudeDev): {dtw['amplitudeDev']} (음수일수록 평소보다 발을 안 드는 패턴)",
        f"- 유사도 타임라인(3초 간격): {dtw['timeline']}",
        "",
        "위 물리적 수치와 DTW 편차를 종합 분석하여 사용자의 러닝 자세를 과학적으로 교정해주세요.",
    ]
    return "\n".join(lines)


_SYSTEM_INSTRUCTION = (
    "당신은 러닝 코칭 전문가입니다. 제공된 센서 통계 수치와 개인 기준선 대비 편차(DTW) 정보를 종합해 "
    "사용자에게 과학적이고 구체적인 교정 동작을 한국어로 조언합니다. "
    "DTW 편차 정보는 절대적 자세 판단이 아니라 '이 사용자가 평소보다 얼마나 흐트러졌는가'를 뜻하므로, "
    "'자세가 나쁘다'가 아니라 '평소보다 흐트러졌다' 혹은 '평소 패턴에서 이탈했다' 식으로 표현하세요. "
    "사용자가 즉시 실행할 수 있는 동작 위주로 2문장 이내로 간결하게 조언하며, 말 끝을 흐리지 말고 마침표('.')로 완전히 끝맺음해야 합니다. "
    "\n\n"
    "★태그 판정 절차 (반드시 아래 순서대로 확인하고, 각 조건에 '뚜렷하게' 해당할 때만 태그를 부여하세요. "
    "애매하거나 지표 하나만 살짝 벗어난 경우는 태그를 부여하지 마세요):\n\n"
    "★중요: jerk_p95, acc_z_std, horizontal_jerk_std는 체중·키·보폭 크기에 따라 사람마다 절대 기준선이 다른 "
    "물리량입니다. 반드시 각 값의 정규화 버전(jerk_p95_norm, acc_z_std_norm, horizontal_jerk_std_norm — 모두 "
    "acc_mag_rms로 나눈 '이 사람의 평균 충격 대비 상대 비율')을 기준으로 판단하고, 원본 절대값은 참고용으로만 "
    "사용하세요. swing_stance_gyro_ratio, front_back_ratio, peak_interval_cv는 이미 비율이라 그대로 사용합니다.\n\n"
    "1단계 — 충격 과다 러닝: jerk_p95_norm이 대략 1.0 이상이거나 acc_z_std_norm이 대략 0.6 이상이면 "
    "지속적인 충격 과다로 판단합니다. 여기에 DTW의 amplitudeDev가 뚜렷한 양수(평소보다 과도하게 무겁게 착지)이면 "
    "더 강한 근거가 됩니다.\n\n"
    "2단계 — 발끌기 러닝 vs 좁은 보폭 러닝 (이 둘은 원인이 다르므로 반드시 아래 기준으로 구분하고, "
    "한쪽 조건만 뚜렷하다면 두 태그를 동시에 선택하지 마세요):\n"
    "  (a) 좁은 보폭 러닝 = '보폭 자체가 짧아 발을 빠르게 굴러야만 하는 문제'. "
    "케이던스(cadence_spm)가 180spm 이상이거나, 케이던스가 160spm을 뚜렷하게 초과하면서 choppy_step_index가 15 이상이면 우선 선택합니다. "
    "이때 수직 가속 범위(acc_z_range)가 반드시 좁을 필요는 없습니다. 빠른 짧은 보폭에서는 swing_stance_gyro_ratio가 낮거나 horizontal_jerk_std_norm이 높게 보일 수 있으므로, "
    "케이던스/choppy 신호가 강하면 이를 발끌기가 아니라 좁은 보폭으로 해석하세요.\n"
    "  (b) 발끌기 러닝 = '다리를 충분히 들어올리지 못하는 문제'. 아래 정규화 지표들을 종합해서 판단하세요 "
    "(하나만 살짝 낮다고 선택하지 말고, 여러 지표가 함께 낮을 때만 선택):\n"
    "    - swing_stance_gyro_ratio가 대략 0.45 미만 (스윙 구간에서 다리 회전이 스탠스 구간과 별 차이 없을 만큼 약함)\n"
    "    - jerk_p95_norm이 대략 0.7 미만 (충격 과다 러닝의 상대 충격 수준에 비해 뚜렷하게 낮음)\n"
    "    - acc_z_std_norm이 대략 0.5 미만이거나, 절대 acc_z_range가 다른 abnormal 구간 대비 눈에 띄게 좁음\n"
    "  충격 과다 러닝/페이스 변화 러닝 지표(jerk_p95_norm, acc_z_std_norm)가 더 뚜렷하게 높으면 그쪽을 우선하고 "
    "'발끌기 러닝'을 추가하지 마세요. "
    "동시에 케이던스(cadence_spm)가 180spm 미만이고 choppy_step_index가 15 미만일 때만 선택합니다. "
    "좁은 보폭 조건이 뚜렷하면 '발끌기 러닝'을 함께 선택하지 마세요.\n\n"
    "3단계 — 페이스 변화 러닝: 아래 중 하나라도 뚜렷하면 선택합니다(모두 만족할 필요 없음):\n"
    "  - 전/후반부 에너지 비율(front_back_ratio)이 1.0에서 뚜렷하게 벗어나거나 에너지 변동계수(acc_mag_cv)가 크다 "
    "(구간 내에서 페이스가 들쭉날쭉하다는 뜻)\n"
    "  - DTW의 trend가 worsening 또는 improving이다 (평소 폼 대비 유사도가 한 방향으로 계속 변하고 있다는 뜻이므로 "
    "그 자체로 페이스/폼이 시간에 따라 변화하고 있다는 강한 근거입니다)\n"
    "  - 유사도 타임라인(timeline)이 뚜렷하게 하락/상승 추세를 보이거나 요동친다\n"
    "  - DTW의 shapeDev가 뚜렷하게 크다 (걸음 형태 자체가 평소 대비 달라지고 있음)\n"
    "다른 태그(충격 과다/발끌기/좁은 보폭)로 이미 충분히 설명되는 변화라면 중복으로 추가하지 않아도 됩니다.\n\n"
    "4단계 — 안정 러닝: 1~3단계의 문제 지표 중 어느 것도 뚜렷하게 관찰되지 않고 DTW status가 stable에 가까울 때만 선택합니다. "
    "'안정 러닝'은 '문제 없음'을 의미하므로 다른 문제 태그와 절대 동시에 선택할 수 없으며, "
    "1~3단계 중 하나라도 조건을 충족했다면 반드시 제외하세요.\n\n"
    "최종적으로 tags는 최소 1개 이상 포함해야 하며, 근거가 애매한 태그를 추측으로 추가하지 말고 "
    "가장 뚜렷하게 관찰되는 문제(들)만 선택하세요."
)


def _call_gemini(prompt: str) -> _CoachingOutput | None:
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.3,
            max_output_tokens=4000,
            response_mime_type="application/json",
            response_schema=_CoachingOutput,
        ),
    )
    parsed = getattr(response, "parsed", None)
    if parsed is None and response.text:
        parsed = _CoachingOutput(**json.loads(response.text))
    return parsed


def _call_openai(prompt: str) -> _CoachingOutput | None:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4000,
        response_format=_CoachingOutput,
    )
    return response.choices[0].message.parsed


def _call_llm(prompt: str) -> tuple[str, list[str]] | None:
    try:
        provider = settings.LLM_PROVIDER.lower()
        if provider == "openai":
            parsed = _call_openai(prompt)
        else:
            parsed = _call_gemini(prompt)

        if parsed is None or not parsed.text:
            return None

        text = parsed.text.strip()
        if text and not text.endswith((".", "!", "?")):
            text += "."  # 말 끝이 흐려진 채로 잘려서 오는 경우에 대한 최소 방어

        tags = [t for t in parsed.tags if t in COACHING_TAGS]
        if not tags:
            logger.warning("LLM이 허용된 태그를 반환하지 않음: raw_tags=%s", parsed.tags)

        return text, tags
    except Exception as e:
        logger.error("LLM 호출 실패: %s", e)
        return None


def generate_coaching(
    channels: dict[str, list[float]],
    ratio: float,
    dtw: dict,
    sample_rate_hz: float = _DEFAULT_SAMPLE_RATE_HZ,
) -> tuple[str, list[str]] | None:
    """
    Returns:
        (text, tags) 정상 처리된 경우.
        None: 센서 구간이 너무 짧거나 실질적인 움직임이 없어 분석이 불가능하거나(InsufficientDataError),
              LLM 호출 자체가 실패한 경우. 두 경우 모두 "판단 불가"이므로 호출측에서 동일하게
              처리해도 무방하지만, 필요하다면 로그의 예외 타입으로 구분 가능하다.
    """
    try:
        stats = extract_stats(channels, sample_rate_hz=sample_rate_hz)
    except InsufficientDataError as e:
        logger.info("코칭 생성 스킵(데이터 부족/정지 상태): %s", e)
        return None

    prompt = _build_prompt(stats, ratio, dtw)
    return _call_llm(prompt)
