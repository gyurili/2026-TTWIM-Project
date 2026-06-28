from pydantic import BaseModel, Field, field_validator
from typing import Optional

# ─────────────────────────────────────────────
# 요청 스키마
# ─────────────────────────────────────────────
class StepRequest(BaseModel):
    """
    1스텝 가속도 데이터 요청 바디
    React Native에서 스텝을 감지할 때마다 이 형태로 POST합니다.
    """
    acc_x: list[float] = Field(..., min_length=10, description="X축 가속도 시계열 (m/s²)")
    acc_y: list[float] = Field(..., min_length=10, description="Y축 가속도 시계열 (m/s²)")
    acc_z: list[float] = Field(..., min_length=10, description="Z축(수직) 가속도 시계열 — 힐스트라이크 핵심 축")
    contact_time: float = Field(..., gt=0, description="발 접지 시간 (ms)")

    @field_validator("acc_x", "acc_y", "acc_z")
    @classmethod
    def check_length_equal(cls, v: list[float]) -> list[float]:
        return v

    @field_validator("contact_time")
    @classmethod
    def check_contact_time(cls, v: float) -> float:
        if not (200 <= v <= 8000):
            raise ValueError(f"contact_time은 200~8000ms 범위여야 합니다. 입력값: {v}ms")
        return v

    def lengths_match(self) -> bool:
        return len(self.acc_x) == len(self.acc_y) == len(self.acc_z)


# ─────────────────────────────────────────────
# 응답 스키마
# ─────────────────────────────────────────────
class StepResponse(BaseModel):
    """
    힐스트라이크 감지 핵심 피드백 응답
    """
    is_heel_strike: bool = Field(..., description="힐스트라이크 여부 (true=힐스트라이크, false=정상)")
    label: str = Field(..., description="감지 결과 레이블 ('heel_strike' | 'normal')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="최종 융합 모델이 판단한 확신도 (0~1)")
    message: str = Field(..., description="사용자에게 가이드할 한국어 피드백 메시지")

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_heel_strike": True,
                "label": "heel_strike",
                "confidence": 0.823,
                "message": "힐스트라이크가 감지되었습니다. 발 앞꿈치로 착지하도록 교정하세요."
            }
        }
    }


# ─────────────────────────────────────────────
# 헬스체크 / 에러 스키마
# ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    model_loaded: bool
    threshold: float
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None