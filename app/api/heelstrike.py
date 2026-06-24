import logging
from typing import List  # 👈 추가
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.heelstrike import (
    StepRequest,
    StepResponse,
    ErrorResponse,
)
from app.services.heelstrike_service import predict, ModelNotLoadedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["힐스트라이크 감지"])

# ─────────────────────────────────────────────
# POST /api/v1/detect (배치 처리 버전)
# ─────────────────────────────────────────────

@router.post(
    "/detect",
    response_model=List[StepResponse],  # 👈 응답 스키마를 List로 변경
    status_code=status.HTTP_200_OK,
    summary="힐스트라이크 배치 감지",
    description=(
        "React Native의 SQLite에서 묶음(Batch)으로 전송된 여러 스텝의 "
        "가속도 시계열 배열 리스트를 받아, 각각의 힐스트라이크 여부 결과를 배열로 반환합니다."
    ),
    responses={
        200: {"description": "모든 스텝 감지 성공"},
        400: {"model": ErrorResponse, "description": "잘못된 입력값"},
        503: {"model": ErrorResponse, "description": "모델 미로드"},
    },
)
async def detect_heel_strike_batch(body: List[StepRequest]) -> List[StepResponse]: # 👈 입력을 List로 변경
    """
    힐스트라이크 묶음 감지 엔드포인트.
    """
    # 1. 빈 배열이 들어온 경우 예외 처리
    if not body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="전송된 스텝 데이터가 비어 있습니다."
        )

    results = []
    
    # 2. 리스트로 들어온 복수 개의 스텝을 순회하며 처리
    for idx, step_data in enumerate(body):
        # 배열 길이 일치 검증 (개별 스텝 내부 검증)
        if not step_data.lengths_match():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"[{idx}번째 스텝] acc_x, acc_y, acc_z 길이가 일치하지 않습니다.",
            )

        try:
            # 서비스 레이어의 predict 함수 호출 (기존 함수 그대로 활용)
            result = predict(step_data)
            results.append(result)

        except ModelNotLoadedError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"[{idx}번째 스텝 오류] {str(e)}",
            )
            
        except Exception as e:
            logger.exception(f"[{idx}번째 스텝] 예상치 못한 추론 오류")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"추론 중 오류가 발생했습니다: {str(e)}",
            )

    # 3. 모든 스텝의 분석 결과 리스트 반환
    logger.info(f"[Batch 완료] 총 {len(results)}개의 스텝 분석 성공")
    return results