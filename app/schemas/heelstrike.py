import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool  # 👈 비동기 오버헤드 방지 스레드풀 워커

from app.schemas.heelstrike import StepRequest, StepResponse, ErrorResponse
from app.services.heelstrike_service import predict, ModelNotLoadedError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["힐스트라이크 감지"])

# ─────────────────────────────────────────────
# POST /api/v1/detect (리액트 네이티브 기존 단건 연동)
# ─────────────────────────────────────────────
@router.post(
    "/detect",
    response_model=StepResponse,  # 👈 단건 응답 유지 (RN 수정 불필요)
    status_code=status.HTTP_200_OK,
    summary="힐스트라이크 실시간 감지 (서버 가속 버전)",
    responses={
        200: {"model": StepResponse, "description": "감지 성공"},
        400: {"model": ErrorResponse, "description": "잘못된 입력값"},
        503: {"model": ErrorResponse, "description": "모델 미로드"},
    },
)
async def detect_heel_strike(body: StepRequest) -> StepResponse:
    """
    React Native 코드를 고치지 않고, 매 초마다 요청을 수용할 수 있도록
    서버 내부 워커 스레드와 DTW 캐시 레이어를 통해 초고속으로 처리합니다.
    """
    if not body.lengths_match():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="입력 축 데이터의 길이가 일치하지 않습니다.",
        )

    try:
        # 💡 [핵심 최적화] CPU 연산이 심한 predict 함수를 
        # FastAPI 이벤트 루프를 막지 않는 워커 스레드풀로 돌려 동시성 속도를 보장합니다.
        result = await run_in_threadpool(predict, body)
        return result

    except ModelNotLoadedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("실시간 추론 중 예상치 못한 에러")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"추론 중 에러 발생: {str(e)}"
        )