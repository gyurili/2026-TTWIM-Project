import json
import logging
import random
from typing import Literal

import numpy as np
from scipy.signal import find_peaks
from google import genai
from langsmith import traceable
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.services.knowledge import get_kb

logger = logging.getLogger(__name__)

ABNORMAL_TAGS: list[str] = ["충격 과다 러닝", "발끌기 러닝", "좁은 보폭 러닝", "페이스 변화 러닝"]

STABLE_TAG = "안정 러닝"
STABLE_COACHING_TEXTS: list[str] = [
    "러닝 자세가 안정적이에요.",
    "좋아요. 지금처럼 편안하게 뛰어보세요.",
    "좋은 자세로 잘 뛰고 있어요.",
    "착지와 보폭이 안정적이에요.",
]


def get_stable_coaching_text() -> str:
    return random.choice(STABLE_COACHING_TEXTS)

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

# --- 고도(기압계) 기반 지형 판정 관련 상수 ---
_ALTITUDE_MIN_READINGS = 3              # 이 개수 이상 쌓여야 오르막/내리막 판정을 시도함
_ALTITUDE_SLOPE_THRESHOLD_M = 1.0       # 처음-마지막 고도 차이가 이 값(m) 이상이어야 경사로 판단


class InsufficientDataError(ValueError):
    """센서 구간이 너무 짧거나(샘플 수 부족) 실질적인 움직임이 없어(정지 상태) 지표를 신뢰할 수 없을 때 발생."""


# 2단계 구조 (이슈: 판정 정확도 + 문구 다양성 동시 확보):
#   1단계 _classify_tag  — 센서 지표만 보고 유형 하나를 고른다. KB 미주입, 저온도.
#   2단계 _generate_phrase — 확정된 유형의 KB(원인·피드백)를 근거로 문구를 만든다. 고온도.
# 한 번의 호출로 판정+문구를 같이 하면 KB 텍스트가 판정 규칙과 경쟁해 오분류가 났다(실측).
class _TagOutput(BaseModel):
    # 구조화 출력 단계에서 4개 유형으로 값 자체를 제한한다 (ABNORMAL_TAGS와 같아야 함).
    # 최대 2개까지 허용: 충격/다리동작 축(1·2단계)에서 하나 + 시간축(3단계, 페이스 변화)에서 하나가
    # 동시에 뚜렷할 수 있어서다. 1·2단계끼리는 프롬프트의 판정 절차 자체가 상호배타이므로 둘 다
    # 나오는 경우는 사실상 없다(강제 검증하지는 않음 — LLM 판단에 맡김).
    tags: list[Literal["충격 과다 러닝", "발끌기 러닝", "좁은 보폭 러닝", "페이스 변화 러닝"]] = Field(
        min_length=1, max_length=2
    )

    @field_validator("tags")
    @classmethod
    def _no_duplicate_tags(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("중복된 태그")
        return v


class _PhraseOutput(BaseModel):
    diagnosis: str  # 진단 문장 1개 (지금 무엇이 어떻게 됐는지)
    action: str     # 교정 문장 1개 (지금 바로 할 동작 하나)


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


def classify_slope(readings: list[float]) -> str | None:
    """
    윈도우 배치마다 쌓인 상대 고도(m) 값들로 오르막/내리막을 판정한다.

    readings가 _ALTITUDE_MIN_READINGS개 미만이면 아직 판단할 근거가 부족하므로 None을 반환한다.
    처음과 마지막 값의 차이가 임계값을 넘을 때만 uphill/downhill로 보고, 그 외엔 flat으로 본다.
    """
    if len(readings) < _ALTITUDE_MIN_READINGS:
        return None

    delta = readings[-1] - readings[0]
    if delta >= _ALTITUDE_SLOPE_THRESHOLD_M:
        return "uphill"
    if delta <= -_ALTITUDE_SLOPE_THRESHOLD_M:
        return "downhill"
    return "flat"


@traceable(name="extract_stats")
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

    # low_motion = acc_z_std < _LOW_MOTION_ACC_Z_STD or step_count < _MIN_STEPS_REQUIRED
    # if low_motion:
    #     raise InsufficientDataError(
    #         f"실질적인 러닝 동작이 감지되지 않음 (acc_z_std={acc_z_std:.4f}, step_count={step_count})"
    #     )

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


_SLOPE_DESC = {
    "uphill": "오르막",
    "downhill": "내리막",
}

def _sensor_block(stats: dict, ratio: float, dtw: dict, slope: str | None) -> list[str]:
    """1·2단계 프롬프트가 공유하는 센서 통계 + DTW 편차 + 지형 블록."""
    lines = [
        "[센서 통계: ML 비정상 판정 구간의 물리적 특징]",
        "1. 충격량 및 일관성 (충격 과다 판별 지표)",
        f" - 95% 최대 충격(impact_score): {stats['impact_score']}g",
        f" - 지속성 관절 충격량(jerk_p95): {stats['jerk_p95']} / 정규화값(jerk_p95_norm): {stats['jerk_p95_norm']}",
        f" - 수직 가속 범위(acc_z_range, p5~p95 기준): {stats['acc_z_range']}g (최대={stats['acc_z_max']}g, 최소={stats['acc_z_min']}g)",
        f" - 수직 충격 변동성(acc_z_std): {stats['acc_z_std']} / 정규화값(acc_z_std_norm): {stats['acc_z_std_norm']}",
        f" - 전체 충격 에너지(acc_mag_rms, 정규화 기준값): {stats['acc_mag_rms']}",
        "",
        "2. 다리 스윙 및 리듬 (발끌기 vs 좁은 보폭 판단 핵심 지표)",
        f" - 다리 스윙 평균 속도(gyro_mag_mean): {stats['gyro_mag_mean']} deg/s",
        f" - 스윙 구간 자이로 에너지(swing_gyro_energy): {stats['swing_gyro_energy']} deg/s",
        f" - 스윙/스탠스 자이로 비율(swing_stance_gyro_ratio): {stats['swing_stance_gyro_ratio']}",
        f" - 수평 마찰 진동(horizontal_jerk_std): {stats['horizontal_jerk_std']} / 정규화값(horizontal_jerk_std_norm): {stats['horizontal_jerk_std_norm']}",
        f" - 수직 가속 하위 10%(acc_z_low_p10): {stats['acc_z_low_p10']}g",
        f" - 걸음 수: {stats['step_count']}보",
        f" - 케이던스(cadence_spm): {stats['cadence_spm']} spm",
        f" - 종종걸음 지수(choppy_step_index, ★핵심 지표): {stats['choppy_step_index']}",
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
    ]
    if slope in _SLOPE_DESC:
        lines += [
            "[지형 변화]",
            f"- 기압계 기반 고도 추정 결과, 최근 러닝은 {_SLOPE_DESC[slope]}으로 판단됩니다.",
            "",
        ]
    return lines


# --- 1단계: 태그 분류 (KB 미주입) ------------------------------------------------ #

# 아래 값은 _CLASSIFY_SYSTEM_INSTRUCTION에 서술된 임곗값과 동일하다(값 동결, 위치만 분리).
# LLM은 산술 비교를 종종 틀리므로 비교 연산은 여기서 파이썬이 수행하고, 결과 해석과
# 최종 유형 판정(여러 신호를 종합하는 판단)은 그대로 LLM에 맡긴다.
_IMPACT_JERK_NORM_TH = 1.0
_IMPACT_ACCZ_STD_TH = 0.6
_SHUFFLE_GYRO_RATIO_TH = 0.45
_SHUFFLE_JERK_NORM_TH = 0.7
_SHUFFLE_ACCZ_STD_TH = 0.5
_CHOPPY_CADENCE_TH = 180.0
_CHOPPY_INDEX_TH = 15.0
_PACE_RATIO_DEVIATION_TH = 0.2
_PACE_VARIATION_TH = 0.2
_PACE_TIMELINE_DEVIATION_TH = 0.15
_PACE_SHAPE_DEVIATION_TH = 2.0


def _threshold_flags(stats: dict) -> list[str]:
    """임곗값 대조 결과만 계산한다. 유형 판정(종합 해석)은 하지 않는다."""

    def mark(label: str, key: str, th: float, op: str) -> str:
        v = stats[key]
        hit = v >= th if op == "이상" else v <= th
        return f" - {label}({key}): {v} / 기준 {th:g} {op} → {'해당' if hit else '미해당'}"

    return [
        "[임곗값 대조 결과 — 아래 해당/미해당은 이미 계산된 값입니다. 다시 계산하지 말고 그대로 "
        "신뢰하되, 여러 지표를 종합해 이번 구간을 가장 잘 설명하는 유형 하나를 고르세요.]",
        mark("충격-순간", "jerk_p95_norm", _IMPACT_JERK_NORM_TH, "이상"),
        mark("충격-산포", "acc_z_std_norm", _IMPACT_ACCZ_STD_TH, "이상"),
        mark("발끌기-스윙", "swing_stance_gyro_ratio", _SHUFFLE_GYRO_RATIO_TH, "이하"),
        mark("발끌기-충격", "jerk_p95_norm", _SHUFFLE_JERK_NORM_TH, "이하"),
        mark("발끌기-산포", "acc_z_std_norm", _SHUFFLE_ACCZ_STD_TH, "이하"),
        mark("좁은보폭-케이던스", "cadence_spm", _CHOPPY_CADENCE_TH, "이상"),
        mark("좁은보폭-종종걸음", "choppy_step_index", _CHOPPY_INDEX_TH, "이상"),
        "",
    ]


def _pace_change_is_notable(stats: dict, dtw: dict) -> bool:
    """페이스 변화 태그를 허용할 만큼 시간적 변화가 뚜렷한지 판정한다."""
    evidence_count = 0
    evidence_count += dtw.get("trend") == "worsening"
    front_back_ratio = float(stats.get("front_back_ratio", 1.0))
    if abs(front_back_ratio - 1.0) >= _PACE_RATIO_DEVIATION_TH:
        evidence_count += 1
    if float(stats.get("acc_mag_cv", 0.0)) >= _PACE_VARIATION_TH:
        evidence_count += 1
    if abs(float(dtw.get("shapeDev", 0.0) or 0.0)) >= _PACE_SHAPE_DEVIATION_TH:
        evidence_count += 1

    timeline = dtw.get("timeline") or []
    if len(timeline) >= 2:
        evidence_count += max(timeline) - min(timeline) >= _PACE_TIMELINE_DEVIATION_TH
    return evidence_count >= 2


def _has_impact_signal(stats: dict) -> bool:
    """충격 과다를 다른 다리 동작 유형보다 우선할 만큼 강한지 확인한다."""
    return (
        float(stats.get("jerk_p95_norm", 0.0)) >= _IMPACT_JERK_NORM_TH
        or float(stats.get("acc_z_std_norm", 0.0)) >= _IMPACT_ACCZ_STD_TH
    )


def _classify_prompt(stats: dict, ratio: float, dtw: dict, slope: str | None) -> str:
    lines = _sensor_block(stats, ratio, dtw, slope)
    lines += _threshold_flags(stats)
    pace_allowed = _pace_change_is_notable(stats, dtw)
    lines.append(
        f"[페이스 변화 태그 허용 여부: {'허용' if pace_allowed else '금지'} — "
        "금지이면 tags에 '페이스 변화 러닝'을 절대 넣지 마세요.]"
    )
    lines.append(
        "위 수치와 DTW 편차를 시스템의 태그 판정 절차에 따라 분석해, tags 에 1~2개를 반환하세요. "
        "더 뚜렷한 근거를 가진 유형을 배열의 첫 번째에 두세요."
    )
    return "\n".join(lines)


_CLASSIFY_SYSTEM_INSTRUCTION = (
    "당신은 러닝 자세 분석 전문가입니다. 제공된 센서 통계 수치와 개인 기준선 대비 편차(DTW)만 근거로, "
    "이미 비정상으로 판정된 러닝 구간이 어떤 유형인지 고르세요. 문구는 만들지 않습니다.\n\n"
    "★결과는 1개 또는 2개까지 가능합니다. 아래 1~3단계를 각각 독립적으로 판단한 뒤, '뚜렷하게' "
    "해당하는 유형만 tags 에 담으세요. 1단계와 2단계는 같은 축(충격·다리 동작)을 보는 절차라 그 "
    "안에서는 원래도 하나만 남습니다(2단계 자체가 (a)/(b) 중 하나를 고르는 절차입니다). 여기에 "
    "3단계(페이스 변화)가 별도로 뚜렷하면 추가해 최대 2개가 됩니다. 근거가 애매한 유형을 억지로 "
    "채우지 말고, 애매하면 가장 뚜렷한 유형 하나만 반환하세요. 2개를 반환할 때는 더 뚜렷한 근거를 "
    "가진 유형을 배열의 첫 번째에 두세요.\n\n"
    "★태그 판정 절차 (반드시 아래 순서대로 확인하고, 각 조건에 '뚜렷하게' 해당할 때만 그 유형을 고르세요):\n\n"
    "★중요: jerk_p95, acc_z_std, horizontal_jerk_std는 체중·키·보폭 크기에 따라 사람마다 절대 기준선이 다른 "
    "물리량입니다. 반드시 각 값의 정규화 버전(jerk_p95_norm, acc_z_std_norm, horizontal_jerk_std_norm — 모두 "
    "acc_mag_rms로 나눈 '이 사람의 평균 충격 대비 상대 비율')을 기준으로 판단하고, 원본 절대값은 참고용으로만 "
    "사용하세요. swing_stance_gyro_ratio, front_back_ratio, peak_interval_cv는 이미 비율이라 그대로 사용합니다.\n\n"
    "1단계 — 충격 과다 러닝: jerk_p95_norm이 1.0 이상이거나 acc_z_std_norm이 0.6 이상이면 "
    "지속적인 충격 과다로 판단합니다. 여기에 DTW의 amplitudeDev가 뚜렷한 양수(평소보다 과도하게 무겁게 착지)이면 "
    "더 강한 근거가 됩니다.\n\n"
    "2단계 — 발끌기 러닝 vs 좁은 보폭 러닝 (이 둘은 원인이 다르므로 반드시 아래 기준으로 구분):\n"
    "  (a) 좁은 보폭 러닝 = '보폭 자체가 짧아 발을 빠르게 굴러야만 하는 문제'. "
    "케이던스(cadence_spm)가 180spm 이상이거나, 케이던스가 160spm을 뚜렷하게 초과하면서 choppy_step_index가 15 이상이면 우선 선택합니다. "
    "빠른 짧은 보폭에서는 swing_stance_gyro_ratio가 낮거나 horizontal_jerk_std_norm이 높게 보일 수 있으므로, "
    "케이던스/choppy 신호가 강하면 이를 발끌기가 아니라 좁은 보폭으로 해석하세요.\n"
    "  (b) 발끌기 러닝 = '다리를 충분히 들어올리지 못하는 문제'. 아래 정규화 지표들이 여러 개 함께 낮을 때만 선택하세요 "
    "(하나만 살짝 낮다고 선택하지 마세요):\n"
    "    - swing_stance_gyro_ratio가 0.45 미만\n"
    "    - jerk_p95_norm이 0.7 미만 (충격 과다 러닝의 상대 충격 수준에 비해 뚜렷하게 낮음)\n"
    "    - acc_z_std_norm이 0.5 미만이거나, 절대 acc_z_range가 다른 abnormal 구간 대비 눈에 띄게 좁음\n"
    "  jerk_p95_norm·acc_z_std_norm이 오히려 높으면 발끌기가 아니라 충격 과다입니다. "
    "케이던스가 180spm 이상이거나 choppy_step_index가 15 이상이면 발끌기가 아니라 좁은 보폭입니다.\n\n"
    "3단계 — 페이스 변화 러닝: [페이스 변화 태그 허용 여부]가 허용일 때만 선택합니다. "
    "아래 중 하나 이상이 뚜렷해야 합니다(모두 만족할 필요 없음):\n"
    "  - 전/후반부 에너지 비율(front_back_ratio)이 1.0에서 뚜렷하게 벗어나거나 에너지 변동계수(acc_mag_cv)가 크다\n"
    "  - DTW의 trend가 worsening이다. improving은 회복 추세이므로 페이스 변화 근거로 사용하지 않는다\n"
    "  - 유사도 타임라인(timeline)이 뚜렷하게 하락/상승 추세를 보이거나 요동친다\n"
    "  - DTW의 shapeDev가 뚜렷하게 크다\n"
    "  단, 충격/케이던스 신호가 함께 뚜렷하면 1·2단계 유형을 우선합니다. 페이스 변화는 그 신호가 약하고 "
    "시간적 변화만 두드러질 때 고릅니다.\n\n"
    "위 1~3단계를 독립적으로 판단해 뚜렷한 유형만 tags 에 1~2개 반환하세요."
)


# --- 2단계: 문구 생성 (확정 태그의 KB 원인·피드백 주입) --------------------------- #

# 확정된 유형 하나에 대해서만 KB(app/data/knowledge_base.json, 항목마다 출처 명시)를 주입한다.
# 1단계에서 유형이 이미 확정됐으므로 '원인' 지식을 넣어도 판정을 흐릴 일이 없다(1단계 프롬프트엔
# KB가 없다). 벡터 검색은 쓰지 않는다 — 유형이 하나로 좁혀졌고 실시간 지연을 최소화해야 하므로.
_KB_CAUSE_DOCS = 3
_KB_FEEDBACK_DOCS = 4


def _kb_lines_for(tags: list[str]) -> list[str]:
    """확정된 태그(1~2개) 각각의 '원인' 지식과, 대표 태그(tags[0])의 '피드백' 지식을 모은다.
    교정 동작은 1개만 나가야 하므로 피드백 지식은 대표 태그 것만 주입한다."""
    kb = get_kb()
    out: list[str] = []

    cause_blocks = []
    for tag in tags:
        cats = kb.get(tag, {})
        # 앞쪽 항목에 쏠리는 걸 막으려고 매 호출마다 순서를 섞는다(문구 다양성).
        causes = random.sample(cats.get("원인", []), min(_KB_CAUSE_DOCS, len(cats.get("원인", []))))
        if causes:
            cause_blocks.append((tag, causes))
    if cause_blocks:
        if len(tags) == 2:
            out.append("[원인 지식 — 진단 문장은 아래 두 유형의 원인을 한 문장 안에 자연스럽게 함께 서술]")
        else:
            out.append("[원인 지식 — 진단 문장은 이 범위 안에서 서술]")
        for tag, causes in cause_blocks:
            if len(tags) == 2:
                out.append(f" [{tag}]")
            out += [f"  - {d['text']}" for d in causes]
        out.append("")

    primary = tags[0]
    acts = random.sample(
        kb.get(primary, {}).get("피드백", []),
        min(_KB_FEEDBACK_DOCS, len(kb.get(primary, {}).get("피드백", []))),
    )
    if acts:
        out.append(f"[교정 지식 — 교정 문장은 이 중 한 가지 동작만 사용 ('{primary}' 기준)]")
        out += [f" - {d['text']}" for d in acts]
        out.append("")
    return out


def _phrase_prompt(tags: list[str], stats: dict, ratio: float, dtw: dict, slope: str | None) -> str:
    lines = _sensor_block(stats, ratio, dtw, slope)
    lines += [f"[확정된 비정상 유형] {' + '.join(tags)}", ""]
    lines += _kb_lines_for(tags)
    if len(tags) == 2:
        lines.append(
            f"이번 구간은 '{tags[0]}'와 '{tags[1]}'이 함께 나타납니다. 진단 문장(diagnosis)에는 두 원인을 "
            f"모두 자연스럽게 담고, 교정 문장(action)은 '{tags[0]}' 기준 동작 하나만 고르세요."
        )
    else:
        lines.append(
            f"이번 수치·편차에서 '{tags[0]}'와 가장 관련 깊은 지표 하나를 골라 그것을 초점으로, "
            "진단 문장(diagnosis)과 교정 문장(action)을 각각 만드세요."
        )
    return "\n".join(lines)


_PHRASE_SYSTEM_INSTRUCTION = (
    "당신은 러닝 코칭 전문가입니다. 비정상 유형은 이미 [확정된 비정상 유형]으로 주어집니다 — 유형을 "
    "바꾸거나 의심하지 말고, 그 유형에 해당하는 현상만 다루세요. 유형이 2개(예: '충격 과다 러닝 + "
    "페이스 변화 러닝')로 주어지면 두 현상이 함께 일어나고 있는 것이니 diagnosis 한 문장 안에 두 "
    "원인을 모두 자연스럽게 담고, action은 그 중 하나(프롬프트가 지정한 대표 유형)의 동작만 고르세요 "
    "— 두 유형의 교정 동작을 섞지 마세요. 사용자가 러닝 중 짧게 듣고 바로 "
    "이해할 수 있는 쉬운 일상어만 쓰고, jerk·cadence·DTW·정규화값 등 지표명이나 전문 용어는 절대 "
    "그대로 언급하지 말고 풀어서 설명하세요.\n"
    "★출력은 두 문장입니다: diagnosis(지금 무엇이 어떻게 됐는지 한 문장) + action(지금 바로 할 동작 한 문장). "
    "유형이 1개면 각 문장을 20~25자 내외로, 유형이 2개라 diagnosis에 원인 두 가지를 담아야 하면 "
    "diagnosis는 35자 내외까지 늘어나도 됩니다. action은 유형 개수와 무관하게 20~25자 내외를 유지하세요. "
    "이유 설명이나 불필요한 수식어 없이 쓰세요.\n"
    "★diagnosis 는 프롬프트 [원인 지식]의 범위 안에서, 이번 수치·편차가 말해 주는 사실을 평서형"
    "(\"~하고 있습니다/합니다\")으로 서술하세요. 명령형·의무형 어미를 쓰지 말고, 근거 없는 원인·수치를 "
    "지어내지 마세요.\n"
    "★action 은 프롬프트 [교정 지식]의 한 가지 동작만 골라 그 의미를 유지한 채 표현·어순·조사만 바꿔 "
    "쓰세요. 목록에 없는 동작을 새로 지어내거나 여러 동작을 섞지 말고, '~보다는'·'~하기보다' 식 대조도 "
    "쓰지 마세요. 어미는 다음 중 상황에 맞는 것을 골라 매번 다르게 끝내세요: "
    "~해보세요 / ~하면 좋습니다 / ~유지하세요 / ~에 집중하세요 / ~를 줄이세요 / ~로 바꾸세요\n"
    "★[교정 지식] 항목이 여러 개면 전부 유효한 선택지입니다. 매번 무난한 하나로 쏠리지 말고, 이번 "
    "수치·편차의 구체적 양상에 가장 잘 맞는 동작을 고르세요. 수치가 다르면 초점도 달라지므로 문장도 "
    "자연히 달라져야 합니다.\n"
    "★'평소보다'라는 표현과 '확실히'는 쓰지 마세요. '기존보다', '기존 패턴에서', '그동안의 흐름과 달리' "
    "등으로 바꾸거나 비교 표현 없이 바로 서술하세요. 감각적 비유(깃털·스프링 등)도 쓰지 마세요.\n"
    "★교정 동작은 즉시 수행 가능하고 부담이 적은 것만 제시하세요. 통증·부상을 단정하거나 진단·치료에 "
    "해당하는 표현을 쓰지 말고, 무리한 강도 변화(급격한 보폭 확대, 속도 증가 등)를 지시하지 마세요."
)

# 프롬프트에 [지형 변화] 섹션이 실제로 있을 때만(오르막/내리막) 붙이는 추가 지시문
_TERRAIN_SYSTEM_ADDENDUM = (
    "★지형 정보: 프롬프트에 [지형 변화] 섹션(오르막/내리막)이 주어지면, 충격 과다나 페이스 변화로 보이는 "
    "지표가 경사로 인한 자연스러운 변화일 수 있음을 감안해 조언 문구에 짧게 반영하세요. "
    "다만 지형과 무관한 판정 기준까지 지형 탓으로 돌리지는 마세요."
)


def _build_system_instruction(base: str, slope: str | None) -> str:
    if slope in _SLOPE_DESC:
        return base + "\n\n" + _TERRAIN_SYSTEM_ADDENDUM
    return base


# 1단계는 판정 재현성을 위해 0으로, 2단계는 문구 다양성을 위해 높게. 한 호출로 둘 다 하던 때의
# 온도 트레이드오프(판정↔다양성)를 단계 분리로 해소한다.
# 주의: temperature=0이 정확도를 높이지는 않는다 — 같은 입력에 같은 답이 나올 확률(재현성)만
# 올라간다. 판정 정확도는 프롬프트의 판정 기준과 입력 피처가 결정한다.
_CLASSIFY_TEMPERATURE = 0.0
_PHRASE_TEMPERATURE = 0.8

# 2단계 LLM 이 실패했을 때 쓰는 태그별 고정 2문장.
_FALLBACK_SENTENCES = {
    "충격 과다 러닝": "착지 충격이 커지고 있습니다. 무릎을 부드럽게 굽혀 흡수해보세요.",
    "발끌기 러닝": "발이 지면에 끌리고 있습니다. 무릎을 가볍게 들어 올려보세요.",
    "좁은 보폭 러닝": "걸음이 잘게 나뉘고 있습니다. 보폭을 자연스럽게 늘려보세요.",
    "페이스 변화 러닝": "러닝 리듬이 흔들리고 있습니다. 호흡을 고르게 맞춰보세요.",
}


def _end_period(s: str) -> str:
    s = s.strip()
    return s if s.endswith((".", "!", "?")) else s + "."


def _call_structured(prompt: str, system_instruction: str, schema: type[BaseModel], temperature: float):
    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai":
        return _call_openai(prompt, system_instruction, schema, temperature)
    return _call_gemini(prompt, system_instruction, schema, temperature)


@traceable(name="call_gemini")
def _call_gemini(prompt: str, system_instruction: str, schema: type[BaseModel], temperature: float):
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=genai.types.HttpOptions(timeout=20_000),
    )
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=256,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    parsed = getattr(response, "parsed", None)
    if parsed is None and response.text:
        parsed = schema(**json.loads(response.text))
    return parsed


@traceable(name="call_openai")
def _call_openai(prompt: str, system_instruction: str, schema: type[BaseModel], temperature: float):
    client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=20.0)
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=256,
        response_format=schema,
    )
    return response.choices[0].message.parsed


@traceable(name="classify_tag")
def _classify_tag(stats: dict, ratio: float, dtw: dict, slope: str | None) -> list[str] | None:
    """1단계: 센서 지표만으로 비정상 유형을 1~2개 고른다. KB 미주입, 저온도."""
    try:
        parsed = _call_structured(
            _classify_prompt(stats, ratio, dtw, slope),
            _build_system_instruction(_CLASSIFY_SYSTEM_INSTRUCTION, slope),
            _TagOutput,
            _CLASSIFY_TEMPERATURE,
        )
    except Exception as e:
        logger.error("태그 분류 LLM 호출 실패: %s", e)
        return None
    if parsed is None or not parsed.tags or any(t not in ABNORMAL_TAGS for t in parsed.tags):
        logger.warning("태그 분류 결과가 유효하지 않음: %s", parsed)
        return None
    if _has_impact_signal(stats) and "발끌기 러닝" in parsed.tags:
        corrected_tags = [
            "충격 과다 러닝" if tag == "발끌기 러닝" else tag
            for tag in parsed.tags
        ]
        logger.info(
            "충격 근거가 발끌기 판정보다 우선하여 태그 보정: stats=%s, tags=%s",
            stats,
            parsed.tags,
        )
        parsed.tags = list(dict.fromkeys(corrected_tags))
    if "페이스 변화 러닝" in parsed.tags:
        physical_tags = [tag for tag in parsed.tags if tag != "페이스 변화 러닝"]
        if physical_tags:
            logger.info("물리적 이상 태그 우선으로 페이스 변화 태그 제거: tags=%s", parsed.tags)
            return physical_tags
        if not _pace_change_is_notable(stats, dtw):
            logger.info("페이스 변화 근거 부족으로 태그 제거: stats=%s, dtw=%s", stats, dtw)
            return None
    return parsed.tags


@traceable(name="generate_phrase")
def _generate_phrase(tags: list[str], stats: dict, ratio: float, dtw: dict, slope: str | None) -> str:
    """2단계: 확정 유형(1~2개)의 KB(원인·피드백)를 근거로 진단·교정 2문장을 만든다. 고온도.
    실패하면 대표 태그(tags[0]) 고정 문장으로 폴백한다."""
    try:
        parsed = _call_structured(
            _phrase_prompt(tags, stats, ratio, dtw, slope),
            _build_system_instruction(_PHRASE_SYSTEM_INSTRUCTION, slope),
            _PhraseOutput,
            _PHRASE_TEMPERATURE,
        )
    except Exception as e:
        logger.error("문구 생성 LLM 호출 실패: %s", e)
        parsed = None
    if parsed is None or not parsed.diagnosis.strip() or not parsed.action.strip():
        logger.warning("문구 생성 결과가 비어 폴백 사용: tags=%s", tags)
        return _FALLBACK_SENTENCES.get(tags[0], "자세가 흐트러지고 있습니다. 리듬을 유지해보세요.")
    return f"{_end_period(parsed.diagnosis)} {_end_period(parsed.action)}"


@traceable(name="generate_coaching")
def generate_coaching(
    channels: dict[str, list[float]],
    ratio: float,
    dtw: dict,
    sample_rate_hz: float = _DEFAULT_SAMPLE_RATE_HZ,
    altitude_readings: list[float] | None = None,
) -> tuple[str, list[str], str | None] | None:
    try:
        stats = extract_stats(channels, sample_rate_hz=sample_rate_hz)
    except InsufficientDataError as e:
        logger.info("코칭 생성 스킵(데이터 부족/정지 상태): %s", e)
        return None

    slope = classify_slope(altitude_readings) if altitude_readings else None

    tags = _classify_tag(stats, ratio, dtw, slope)  # 1단계
    if tags is None:
        logger.info("코칭 생성 스킵(태그 분류 실패)")
        return None

    text = _generate_phrase(tags, stats, ratio, dtw, slope)  # 2단계
    logger.info("중간피드백 생성: tags=%s, slope=%s, text=%s", tags, slope, text)
    return text, tags, slope
