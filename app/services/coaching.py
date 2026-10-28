
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

ABNORMAL_TAGS: list[str] = [
    "충격 과다 러닝",
    "발끌기 러닝",
    "좁은 보폭 러닝",
    "페이스 변화 러닝",
]

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
_DEFAULT_SAMPLE_RATE_HZ = 50.0
_MIN_SECONDS_REQUIRED = 1.5
_MIN_STEPS_REQUIRED = 2
_LOW_MOTION_ACC_Z_STD = 0.05
_MIN_GYRO_FLOOR = 1e-3
_MIN_STEP_DISTANCE_SECONDS = 0.25
_STEP_PROMINENCE_RATIO = 0.12
_MIN_STEP_PROMINENCE = 0.15
_CHOPPY_BASE_CADENCE_SPM = 160.0

# --- 고도(기압계) 기반 지형 판정 관련 상수 ---
_ALTITUDE_MIN_READINGS = 3
_ALTITUDE_SLOPE_THRESHOLD_M = 1.0


class InsufficientDataError(ValueError):
    """센서 구간이 너무 짧거나 유효한 움직임이 부족할 때 발생."""


class _TagOutput(BaseModel):
    tags: list[
        Literal[
            "충격 과다 러닝",
            "발끌기 러닝",
            "좁은 보폭 러닝",
            "페이스 변화 러닝",
        ]
    ] = Field(min_length=1, max_length=2)

    @field_validator("tags")
    @classmethod
    def _no_duplicate_tags(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("중복된 태그")
        return v


class _PhraseOutput(BaseModel):
    diagnosis: str
    action: str


def _safe_div(
    numerator: float,
    denominator: float,
    floor: float = _EPS,
) -> float:
    denom = (
        denominator
        if abs(denominator) >= floor
        else (floor if denominator >= 0 else -floor)
    )
    return float(numerator / denom)


def _finite_round(x: float, nd: int = 4) -> float:
    x = float(x)
    if not np.isfinite(x):
        return 0.0
    return round(x, nd)


def classify_slope(readings: list[float]) -> str | None:
    """상대 고도 변화로 오르막/내리막을 판정한다."""
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
    """센서 채널에서 러닝 폼 분석용 피처를 계산한다."""

    required = (
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    )

    missing = [k for k in required if k not in channels]
    if missing:
        raise InsufficientDataError(f"필수 채널 누락: {missing}")

    acc_x = np.asarray(channels["acc_x"], dtype=float)
    acc_y = np.asarray(channels["acc_y"], dtype=float)
    acc_z = np.asarray(channels["acc_z"], dtype=float)
    gyro_x = np.asarray(channels["gyro_x"], dtype=float)
    gyro_y = np.asarray(channels["gyro_y"], dtype=float)
    gyro_z = np.asarray(channels["gyro_z"], dtype=float)

    lengths = {
        len(acc_x),
        len(acc_y),
        len(acc_z),
        len(gyro_x),
        len(gyro_y),
        len(gyro_z),
    }

    if len(lengths) != 1:
        raise InsufficientDataError(
            f"채널 간 길이가 서로 다름: {lengths}"
        )

    total_frames = len(acc_z)
    min_frames_required = max(
        10,
        int(sample_rate_hz * _MIN_SECONDS_REQUIRED),
    )

    if total_frames < min_frames_required:
        raise InsufficientDataError(
            f"프레임 수 부족: {total_frames} < {min_frames_required} "
            f"(최소 {_MIN_SECONDS_REQUIRED}초 필요)"
        )

    all_channels = np.stack(
        [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z],
        axis=1,
    )

    if not np.all(np.isfinite(all_channels)):
        raise InsufficientDataError(
            "입력 채널에 NaN/Inf 값이 포함되어 있음"
        )

    gyro = np.stack([gyro_x, gyro_y, gyro_z], axis=1)
    acc = np.stack([acc_x, acc_y, acc_z], axis=1)

    acc_mag = np.sqrt(np.sum(acc**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))

    # 1. 충격량 및 일관성 분석
    acc_z_mean = float(np.mean(acc_z))
    acc_z_max = float(np.max(acc_z))
    acc_z_min = float(np.min(acc_z))

    acc_z_range_robust = float(
        np.percentile(acc_z, 95) - np.percentile(acc_z, 5)
    )

    acc_z_std = float(np.std(acc_z))
    acc_mag_rms = float(np.sqrt(np.mean(acc_mag**2)))
    impact_score = float(np.percentile(acc_mag, 95))

    jerk = np.abs(np.diff(acc_mag))
    jerk_p95 = (
        float(np.percentile(jerk, 95))
        if len(jerk) > 0
        else 0.0
    )

    # 2. 다리 스윙 및 지면 마찰 분석
    gyro_mag_mean = float(np.mean(gyro_mag))

    swing_mask = acc_mag < np.median(acc_mag)
    stance_mask = ~swing_mask

    swing_gyro_energy = (
        float(np.mean(gyro_mag[swing_mask]))
        if np.any(swing_mask)
        else gyro_mag_mean
    )

    stance_gyro_energy = (
        float(np.mean(gyro_mag[stance_mask]))
        if np.any(stance_mask)
        else gyro_mag_mean
    )

    swing_stance_gyro_ratio = _safe_div(
        swing_gyro_energy,
        stance_gyro_energy,
        floor=_MIN_GYRO_FLOOR,
    )

    jerk_x = np.diff(acc_x)
    jerk_y = np.diff(acc_y)

    horizontal_jerk_std = (
        float(np.std(np.sqrt(jerk_x**2 + jerk_y**2)))
        if len(jerk_x) > 0
        else 0.0
    )

    acc_z_low_p10 = float(np.percentile(acc_z, 10))
    acc_z_centered = acc_z - acc_z_mean
    duration_sec = total_frames / sample_rate_hz

    # 3. 보폭 및 케이던스 분석
    min_peak_distance = max(
        1,
        int(sample_rate_hz * _MIN_STEP_DISTANCE_SECONDS),
    )

    peak_prominence = max(
        _MIN_STEP_PROMINENCE,
        acc_z_range_robust * _STEP_PROMINENCE_RATIO,
    )

    step_peaks, _ = find_peaks(
        acc_z_centered,
        distance=min_peak_distance,
        prominence=peak_prominence,
    )

    step_count = int(len(step_peaks))

    cadence_spm = (
        float(step_count / duration_sec * 60.0)
        if duration_sec > 0
        else 0.0
    )

    if len(step_peaks) > 2:
        intervals = np.diff(step_peaks)
        peak_interval_cv = float(
            np.std(intervals) / (np.mean(intervals) + _EPS)
        )
    else:
        peak_interval_cv = 0.0

    cadence_excess = max(
        0.0,
        cadence_spm - _CHOPPY_BASE_CADENCE_SPM,
    )

    rhythm_regularity_factor = _safe_div(
        1.0,
        1.0 + peak_interval_cv,
    )

    choppy_step_index = (
        cadence_excess * rhythm_regularity_factor
    )

    autocorr = np.correlate(
        acc_z_centered,
        acc_z_centered,
        mode="full",
    )

    autocorr = autocorr[len(acc_z_centered) - 1:]
    stride_regularity = 0.0

    if len(autocorr) > 1 and autocorr[0] > 0:
        min_lag = max(1, int(0.2 * sample_rate_hz))
        max_lag = min(
            len(autocorr) - 1,
            int(2.0 * sample_rate_hz),
        )

        if max_lag > min_lag:
            window = autocorr[min_lag:max_lag]
            best_lag_value = float(np.max(window))

            stride_regularity = max(
                0.0,
                min(1.0, best_lag_value / autocorr[0]),
            )

    # 4. 페이스 및 시간적 변화 분석
    acc_mag_mean = float(np.mean(acc_mag))

    acc_mag_cv = _safe_div(
        float(np.std(acc_mag)),
        acc_mag_mean,
        floor=_EPS,
    )

    N = len(acc_mag)

    if N > 10:
        back_energy = float(np.mean(acc_mag[N // 2:] ** 2))

        front_back_ratio = _safe_div(
            float(np.mean(acc_mag[:N // 2] ** 2)),
            back_energy,
            floor=_EPS,
        )
    else:
        front_back_ratio = 1.0

    # 정규화 지표
    jerk_p95_norm = _safe_div(jerk_p95, acc_mag_rms)
    acc_z_std_norm = _safe_div(acc_z_std, acc_mag_rms)

    horizontal_jerk_std_norm = _safe_div(
        horizontal_jerk_std,
        acc_mag_rms,
    )

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
        "horizontal_jerk_std": _finite_round(
            horizontal_jerk_std
        ),
        "horizontal_jerk_std_norm": _finite_round(
            horizontal_jerk_std_norm
        ),
        "acc_mag_rms": _finite_round(acc_mag_rms),
        "gyro_mag_mean": _finite_round(gyro_mag_mean),
        "swing_gyro_energy": _finite_round(swing_gyro_energy),
        "swing_stance_gyro_ratio": _finite_round(
            swing_stance_gyro_ratio
        ),
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


def _sensor_block(
    stats: dict,
    ratio: float,
    dtw: dict,
    slope: str | None,
) -> list[str]:
    """센서 통계, DTW 편차 및 지형 정보를 구성한다."""

    lines = [
        "[센서 통계: ML 비정상 판정 구간의 물리적 특징]",
        "1. 충격량 및 일관성 (충격 과다 판별 지표)",
        f" - 95% 최대 충격(impact_score): {stats['impact_score']}g",
        f" - 지속성 관절 충격량(jerk_p95): {stats['jerk_p95']} / "
        f"정규화값(jerk_p95_norm): {stats['jerk_p95_norm']}",
        f" - 수직 가속 범위(acc_z_range, p5~p95 기준): "
        f"{stats['acc_z_range']}g "
        f"(최대={stats['acc_z_max']}g, 최소={stats['acc_z_min']}g)",
        f" - 수직 충격 변동성(acc_z_std): {stats['acc_z_std']} / "
        f"정규화값(acc_z_std_norm): {stats['acc_z_std_norm']}",
        f" - 전체 충격 에너지(acc_mag_rms): {stats['acc_mag_rms']}",
        "",
        "2. 다리 스윙 및 리듬 (발끌기 vs 좁은 보폭 판단 지표)",
        f" - 다리 스윙 평균 속도(gyro_mag_mean): "
        f"{stats['gyro_mag_mean']} deg/s",
        f" - 스윙 구간 자이로 에너지(swing_gyro_energy): "
        f"{stats['swing_gyro_energy']} deg/s",
        f" - 스윙/스탠스 자이로 비율(swing_stance_gyro_ratio): "
        f"{stats['swing_stance_gyro_ratio']}",
        f" - 수평 마찰 진동(horizontal_jerk_std): "
        f"{stats['horizontal_jerk_std']} / "
        f"정규화값(horizontal_jerk_std_norm): "
        f"{stats['horizontal_jerk_std_norm']}",
        f" - 수직 가속 하위 10%(acc_z_low_p10): "
        f"{stats['acc_z_low_p10']}g",
        f" - 걸음 수: {stats['step_count']}보",
        f" - 케이던스(cadence_spm): {stats['cadence_spm']} spm",
        f" - 종종걸음 지수(choppy_step_index): "
        f"{stats['choppy_step_index']}",
        f" - 보폭 간격 변동계수(peak_interval_cv): "
        f"{stats['peak_interval_cv']}",
        f" - 보폭 규칙성(stride_regularity): "
        f"{stats['stride_regularity']}",
        "",
        "3. 페이스 및 시간적 변화",
        f" - 에너지 변동계수(acc_mag_cv): {stats['acc_mag_cv']}",
        f" - 전/후반부 에너지 비율(front_back_ratio): "
        f"{stats['front_back_ratio']}",
        f" - 구간 내 비정상 비율: {ratio:.0%}",
        "",
        "[개인 기준선 대비 편차 (DTW)]",
        f"- 상태: {dtw['status']} "
        f"({_DTW_STATUS_DESC.get(dtw['status'], dtw['status'])})",
        f"- 평균 유사도: {dtw['avgSimilarity']}",
        f"- 추세: {dtw['trend']} "
        f"({_DTW_TREND_DESC.get(dtw['trend'], dtw['trend'])})",
        f"- 걸음 형태 편차(shapeDev): {dtw['shapeDev']}",
        f"- 속도 편차(cadenceDev): {dtw['cadenceDev']}",
        f"- 발 움직임 편차(amplitudeDev): {dtw['amplitudeDev']}",
        f"- 유사도 타임라인(3초 간격): {dtw['timeline']}",
        "",
    ]

    if slope in _SLOPE_DESC:
        lines += [
            "[지형 변화]",
            f"- 기압계 기반 고도 추정 결과, 최근 러닝은 "
            f"{_SLOPE_DESC[slope]}으로 판단됩니다.",
            "",
        ]

    return lines


# --- 1단계: 태그 분류 ------------------------------------------------------------

_IMPACT_JERK_NORM_TH = 0.85
_IMPACT_ACCZ_STD_TH = 0.55
_IMPACT_TO_RMS_RATIO_TH = 1.6

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
    """계산된 임곗값 비교 결과를 프롬프트에 전달한다."""

    def mark(label: str, key: str, th: float, op: str) -> str:
        v = stats[key]
        hit = v >= th if op == "이상" else v <= th

        return (
            f" - {label}({key}): {v} / 기준 {th:g} {op} "
            f"→ {'해당' if hit else '미해당'}"
        )

    return [
        "[임곗값 대조 결과 — 아래 결과를 다시 계산하지 말고 "
        "여러 지표를 종합해 유형을 판단하세요.]",
        mark(
            "충격-순간",
            "jerk_p95_norm",
            _IMPACT_JERK_NORM_TH,
            "이상",
        ),
        mark(
            "충격-산포",
            "acc_z_std_norm",
            _IMPACT_ACCZ_STD_TH,
            "이상",
        ),
        mark(
            "발끌기-스윙",
            "swing_stance_gyro_ratio",
            _SHUFFLE_GYRO_RATIO_TH,
            "이하",
        ),
        mark(
            "발끌기-충격",
            "jerk_p95_norm",
            _SHUFFLE_JERK_NORM_TH,
            "이하",
        ),
        mark(
            "발끌기-산포",
            "acc_z_std_norm",
            _SHUFFLE_ACCZ_STD_TH,
            "이하",
        ),
        mark(
            "좁은보폭-케이던스",
            "cadence_spm",
            _CHOPPY_CADENCE_TH,
            "이상",
        ),
        mark(
            "좁은보폭-종종걸음",
            "choppy_step_index",
            _CHOPPY_INDEX_TH,
            "이상",
        ),
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
        evidence_count += (
            max(timeline) - min(timeline)
            >= _PACE_TIMELINE_DEVIATION_TH
        )

    return evidence_count >= 2


def _has_impact_signal(stats: dict) -> bool:
    """
    충격 과다 신호를 우선 확인한다.

    주의: 완화된 임곗값은 오분류를 줄이기 위한 초기 조정값이며,
    실제 데이터셋에서 검증된 최적 임곗값은 아니다.
    """

    jerk_norm = float(stats.get("jerk_p95_norm", 0.0))
    acc_std_norm = float(stats.get("acc_z_std_norm", 0.0))
    impact_score = float(stats.get("impact_score", 0.0))
    acc_mag_rms = float(stats.get("acc_mag_rms", 0.0))

    impact_to_rms_ratio = _safe_div(
        impact_score,
        acc_mag_rms,
    )

    return (
        jerk_norm >= _IMPACT_JERK_NORM_TH
        or acc_std_norm >= _IMPACT_ACCZ_STD_TH
        or impact_to_rms_ratio >= _IMPACT_TO_RMS_RATIO_TH
    )


def _classify_prompt(
    stats: dict,
    ratio: float,
    dtw: dict,
    slope: str | None,
) -> str:
    lines = _sensor_block(stats, ratio, dtw, slope)
    lines += _threshold_flags(stats)

    pace_allowed = _pace_change_is_notable(stats, dtw)

    lines.append(
        f"[페이스 변화 태그 허용 여부: "
        f"{'허용' if pace_allowed else '금지'} — "
        "금지이면 tags에 '페이스 변화 러닝'을 넣지 마세요.]"
    )

    lines.append(
        "위 수치와 DTW 편차를 판정 절차에 따라 분석해 "
        "tags에 1~2개를 반환하세요. "
        "더 뚜렷한 근거를 가진 유형을 첫 번째에 두세요."
    )

    return "\n".join(lines)


_CLASSIFY_SYSTEM_INSTRUCTION = (
    "당신은 러닝 자세 분석 전문가입니다. 제공된 센서 통계와 "
    "개인 기준선 대비 편차(DTW)만으로 비정상 구간의 유형을 고르세요. "
    "문구는 만들지 않습니다.\n\n"

    "결과는 1개 또는 2개입니다. 근거가 애매한 유형을 억지로 "
    "채우지 마세요. 1단계와 2단계는 충격·다리 동작을 판단하고, "
    "3단계는 시간적 변화를 별도로 판단합니다.\n\n"

    "★중요: jerk_p95, acc_z_std, horizontal_jerk_std는 "
    "절대값보다 정규화 버전인 jerk_p95_norm, acc_z_std_norm, "
    "horizontal_jerk_std_norm을 우선 참고하세요.\n\n"

    
"1단계 — 충격 과다 러닝:\n"
"jerk_p95_norm, acc_z_std_norm, impact_score / acc_mag_rms를 "
"함께 확인하세요. 하나의 지표만 기준을 넘었다는 이유로 "
"충격 과다를 확정하지 마세요. "
"여러 충격 지표가 동시에 높을 때 충격 과다를 우선 검토하세요.\n\n"

"★발끌기와 충격 과다 구분:\n"
"발끌기는 다리 스윙이 부족하고 발이 지면에 끌리는 패턴입니다. "
"충격 과다는 착지 순간의 충격과 가속도 변화가 크게 나타나는 패턴입니다. "
"스윙/스탠스 자이로 비율이 낮고 충격 지표가 충분히 높지 않다면 "
"발끌기를 우선하세요. "
"충격 지표 여러 개가 함께 높을 때만 충격 과다를 우선하세요.\n\n"

    "2단계 — 발끌기 러닝 vs 좁은 보폭 러닝:\n"
    "(a) 좁은 보폭 러닝은 보폭이 짧고 발을 빠르게 굴리는 패턴입니다. "
    "케이던스가 180spm 이상이거나 케이던스가 160spm을 뚜렷하게 "
    "초과하면서 choppy_step_index가 15 이상이면 좁은 보폭을 우선합니다. "
    "케이던스 신호가 강하면 발끌기보다 좁은 보폭으로 해석하세요.\n\n"

    "(b) 발끌기 러닝은 다리를 충분히 들어 올리지 못하는 패턴입니다. "
    "swing_stance_gyro_ratio < 0.45, jerk_p95_norm < 0.7, "
    "acc_z_std_norm < 0.5 또는 수직 가속 범위가 다른 비정상 구간보다 "
    "눈에 띄게 좁다는 근거를 종합하세요. "
    "하나의 지표만 낮다고 발끌기로 선택하지 마세요.\n"
    "충격 과다 신호가 강하면 발끌기보다 충격 과다를 우선하세요. "
    "케이던스가 180spm 이상이거나 choppy_step_index가 15 이상이면 "
    "좁은 보폭 여부를 우선 검토하세요.\n\n"

    "3단계 — 페이스 변화 러닝:\n"
    "[페이스 변화 태그 허용 여부]가 허용일 때만 선택하세요. "
    "front_back_ratio가 1.0에서 크게 벗어나거나 acc_mag_cv가 크거나, "
    "DTW trend가 worsening이거나 timeline이 크게 변하거나 "
    "shapeDev가 큰 경우를 근거로 삼으세요. "
    "충격 신호가 강하면 충격 과다를 우선하세요.\n\n"

    "위 절차에 따라 근거가 뚜렷한 유형만 tags에 1~2개 반환하세요."
)


# --- 2단계: 문구 생성 ------------------------------------------------------------

_KB_CAUSE_DOCS = 3
_KB_FEEDBACK_DOCS = 4


def _kb_lines_for(tags: list[str]) -> list[str]:
    """확정 태그에 해당하는 원인 및 대표 태그의 피드백을 구성한다."""

    kb = get_kb()
    out: list[str] = []
    cause_blocks = []

    for tag in tags:
        cats = kb.get(tag, {})
        causes = random.sample(
            cats.get("원인", []),
            min(_KB_CAUSE_DOCS, len(cats.get("원인", []))),
        )

        if causes:
            cause_blocks.append((tag, causes))

    if cause_blocks:
        if len(tags) == 2:
            out.append(
                "[원인 지식 — 두 유형의 원인을 진단 문장에 함께 서술]"
            )
        else:
            out.append("[원인 지식 — 진단 문장은 이 범위 안에서 서술]")

        for tag, causes in cause_blocks:
            if len(tags) == 2:
                out.append(f" [{tag}]")

            out.extend(
                f"  - {doc['text']}"
                for doc in causes
            )

        out.append("")

    primary = tags[0]
    acts = random.sample(
        kb.get(primary, {}).get("피드백", []),
        min(
            _KB_FEEDBACK_DOCS,
            len(kb.get(primary, {}).get("피드백", [])),
        ),
    )

    if acts:
        out.append(
            f"[교정 지식 — '{primary}' 기준으로 동작 하나만 선택]"
        )
        out.extend(f" - {doc['text']}" for doc in acts)
        out.append("")

    return out


def _phrase_prompt(
    tags: list[str],
    stats: dict,
    ratio: float,
    dtw: dict,
    slope: str | None,
) -> str:
    lines = _sensor_block(stats, ratio, dtw, slope)
    lines += [f"[확정된 비정상 유형] {' + '.join(tags)}", ""]
    lines += _kb_lines_for(tags)

    if len(tags) == 2:
        lines.append(
            f"이번 구간은 '{tags[0]}'와 '{tags[1]}'이 함께 나타납니다. "
            "diagnosis에는 두 현상을 담고, action은 대표 유형의 "
            "교정 동작 하나만 선택하세요."
        )
    else:
        lines.append(
            f"이번 구간에서 '{tags[0]}'와 가장 관련 깊은 지표 하나를 "
            "선택해 diagnosis와 action을 만드세요."
        )

    return "\n".join(lines)


_PHRASE_SYSTEM_INSTRUCTION = (
    "당신은 러닝 코칭 전문가입니다. 비정상 유형은 이미 확정되어 있습니다. "
    "유형을 바꾸지 말고 해당 현상만 다루세요. 두 유형이 주어지면 "
    "diagnosis에는 두 현상을 담고 action은 대표 유형의 동작 하나만 고르세요. "
    "러닝 중 바로 이해할 수 있는 쉬운 일상어를 사용하고 전문 지표명은 "
    "그대로 언급하지 마세요.\n\n"

    "출력은 diagnosis와 action 두 문장입니다. "
    "유형이 하나면 각 문장은 20~25자 내외, 두 유형이면 diagnosis는 "
    "35자 내외까지 허용합니다. action은 20~25자 내외로 유지하세요.\n\n"

    "diagnosis는 원인 지식의 범위 안에서 현재 상태를 평서형으로 설명하세요. "
    "근거 없는 원인이나 수치를 만들지 마세요.\n\n"

    "action은 교정 지식에서 한 가지 동작만 선택하세요. "
    "목록에 없는 동작을 새로 만들거나 여러 동작을 섞지 마세요. "
    "어미는 '~해보세요', '~하면 좋습니다', '~유지하세요', "
    "'~에 집중하세요', '~를 줄이세요', '~로 바꾸세요' 등을 사용하세요.\n\n"

    "'평소보다'와 '확실히'는 쓰지 마세요. "
    "'기존보다', '기존 패턴에서' 등으로 바꾸거나 비교 표현 없이 서술하세요. "
    "감각적 비유를 쓰지 마세요.\n\n"

    "즉시 수행 가능하고 부담이 적은 동작만 제시하세요. "
    "통증이나 부상을 단정하거나 진단·치료 표현을 사용하지 마세요. "
    "급격한 보폭 확대나 속도 증가를 지시하지 마세요."
)


_TERRAIN_SYSTEM_ADDENDUM = (
    "지형 정보: [지형 변화] 섹션이 있으면 충격이나 페이스 변화가 "
    "경사로 인한 자연스러운 변화일 수 있음을 고려해 조언에 짧게 반영하세요. "
    "단, 지형과 무관한 판정 기준까지 지형 탓으로 돌리지 마세요."
)


def _build_system_instruction(
    base: str,
    slope: str | None,
) -> str:
    if slope in _SLOPE_DESC:
        return base + "\n\n" + _TERRAIN_SYSTEM_ADDENDUM
    return base


_CLASSIFY_TEMPERATURE = 0.0
_PHRASE_TEMPERATURE = 0.8


_FALLBACK_SENTENCES = {
    "충격 과다 러닝": (
        "착지 충격이 커지고 있습니다. "
        "무릎을 부드럽게 굽혀 흡수해보세요."
    ),
    "발끌기 러닝": (
        "발이 지면에 끌리고 있습니다. "
        "무릎을 가볍게 들어 올려보세요."
    ),
    "좁은 보폭 러닝": (
        "걸음이 잘게 나뉘고 있습니다. "
        "보폭을 자연스럽게 늘려보세요."
    ),
    "페이스 변화 러닝": (
        "러닝 리듬이 흔들리고 있습니다. "
        "호흡을 고르게 맞춰보세요."
    ),
}


def _end_period(s: str) -> str:
    s = s.strip()
    return s if s.endswith((".", "!", "?")) else s + "."


def _call_structured(
    prompt: str,
    system_instruction: str,
    schema: type[BaseModel],
    temperature: float,
):
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        return _call_openai(
            prompt,
            system_instruction,
            schema,
            temperature,
        )

    return _call_gemini(
        prompt,
        system_instruction,
        schema,
        temperature,
    )


@traceable(name="call_gemini")
def _call_gemini(
    prompt: str,
    system_instruction: str,
    schema: type[BaseModel],
    temperature: float,
):
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
def _call_openai(
    prompt: str,
    system_instruction: str,
    schema: type[BaseModel],
    temperature: float,
):
    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=20.0,
    )

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        max_tokens=256,
        response_format=schema,
    )

    return response.choices[0].message.parsed


@traceable(name="classify_tag")
def _classify_tag(
    stats: dict,
    ratio: float,
    dtw: dict,
    slope: str | None,
) -> list[str] | None:
    """1단계: 러닝 유형을 분류하고 신호를 기준으로 보정한다."""

    try:
        parsed = _call_structured(
            _classify_prompt(stats, ratio, dtw, slope),
            _build_system_instruction(
                _CLASSIFY_SYSTEM_INSTRUCTION,
                slope,
            ),
            _TagOutput,
            _CLASSIFY_TEMPERATURE,
        )
    except Exception as e:
        logger.error("유형 분류 LLM 호출 실패: %s", e)
        return None

    if parsed is None or not parsed.tags:
        logger.warning("유형 분류 결과가 비어 있음")
        return None

    # 발끌기 관련 지표 확인
    swing_ratio = float(
        stats.get("swing_stance_gyro_ratio", 1.0)
    )
    jerk_norm = float(stats.get("jerk_p95_norm", 0.0))
    acc_std_norm = float(stats.get("acc_z_std_norm", 0.0))

    dragging_signals = sum([
        swing_ratio < 0.45,
        jerk_norm < 0.7,
        acc_std_norm < 0.5,
    ])
    dragging_detected = dragging_signals >= 2

    # 충격 관련 지표는 세 개 중 두 개 이상이 높을 때만 인정
    impact_detected = _has_impact_signal(stats)

    original_tags = list(parsed.tags)

    if dragging_detected:
        corrected_tags = ["발끌기 러닝"]

    elif impact_detected:
        corrected_tags = ["충격 과다 러닝"]

        if "좁은 보폭 러닝" in original_tags:
            corrected_tags.append("좁은 보폭 러닝")

        if (
            "페이스 변화 러닝" in original_tags
            and _pace_change_is_notable(stats, dtw)
        ):
            corrected_tags.append("페이스 변화 러닝")

    else:
        corrected_tags = original_tags

    if corrected_tags != original_tags:
        logger.info(
            "러닝 유형 보정: original=%s, corrected=%s, "
            "swing_ratio=%.3f, jerk_norm=%.3f, "
            "acc_std_norm=%.3f, impact_detected=%s",
            original_tags,
            corrected_tags,
            swing_ratio,
            jerk_norm,
            acc_std_norm,
            impact_detected,
        )

    return corrected_tags

    # 수정 핵심: 충격 신호가 있으면 충격 과다를 대표 태그로 강제한다.
    
def _has_impact_signal(stats: dict) -> bool:
    """여러 충격 지표가 함께 높을 때만 충격 과다로 판정한다."""
    jerk_norm = float(stats.get("jerk_p95_norm", 0.0))
    acc_std_norm = float(stats.get("acc_z_std_norm", 0.0))
    impact_score = float(stats.get("impact_score", 0.0))
    acc_mag_rms = float(stats.get("acc_mag_rms", 0.0))

    impact_ratio = _safe_div(impact_score, acc_mag_rms)

    impact_signals = sum([
        jerk_norm >= 1.0,
        acc_std_norm >= 0.6,
        impact_ratio >= 1.8,
    ])

    # 세 지표 중 적어도 두 개가 높을 때만 True
    return impact_signals >= 2

@traceable(name="generate_phrase")
def _generate_phrase(
    tags: list[str],
    stats: dict,
    ratio: float,
    dtw: dict,
    slope: str | None,
) -> str:
    """2단계: 확정 태그의 지식 베이스를 활용해 코칭 문구를 생성한다."""

    try:
        parsed = _call_structured(
            _phrase_prompt(tags, stats, ratio, dtw, slope),
            _build_system_instruction(
                _PHRASE_SYSTEM_INSTRUCTION,
                slope,
            ),
            _PhraseOutput,
            _PHRASE_TEMPERATURE,
        )
    except Exception as e:
        logger.error("문구 생성 LLM 호출 실패: %s", e)
        parsed = None

    if (
        parsed is None
        or not parsed.diagnosis.strip()
        or not parsed.action.strip()
    ):
        logger.warning(
            "문구 생성 결과가 비어 폴백 사용: tags=%s",
            tags,
        )
        return _FALLBACK_SENTENCES.get(
            tags[0],
            "자세가 흐트러지고 있습니다. 리듬을 유지해보세요.",
        )

    return (
        f"{_end_period(parsed.diagnosis)} "
        f"{_end_period(parsed.action)}"
    )


@traceable(name="generate_coaching")
def generate_coaching(
    channels: dict[str, list[float]],
    ratio: float,
    dtw: dict,
    sample_rate_hz: float = _DEFAULT_SAMPLE_RATE_HZ,
    altitude_readings: list[float] | None = None,
) -> tuple[str, list[str], str | None] | None:
    """러닝 구간을 분석해 코칭 문구, 유형 태그, 지형 정보를 반환한다."""

    try:
        stats = extract_stats(
            channels,
            sample_rate_hz=sample_rate_hz,
        )
    except InsufficientDataError as e:
        logger.info(
            "코칭 생성 스킵(데이터 부족/정지 상태): %s",
            e,
        )
        return None

    slope = (
        classify_slope(altitude_readings)
        if altitude_readings
        else None
    )

    tags = _classify_tag(
        stats,
        ratio,
        dtw,
        slope,
    )

    if tags is None:
        logger.info("코칭 생성 스킵(태그 분류 실패)")
        return None

    text = _generate_phrase(
        tags,
        stats,
        ratio,
        dtw,
        slope,
    )

    logger.info(
        "중간피드백 생성: tags=%s, slope=%s, text=%s",
        tags,
        slope,
        text,
    )

    return text, tags, slope
