import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.db.session import engine
from app.db.base import Base
import app.models.imu  # noqa: F401 - Base에 모델 등록

# 힐스트라이크 모델 로더 임포트
from app.services.heelstrike_service import load_model

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Lifespan: 앱 시작과 종료 시점의 로직 관리
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [1] 서버 시작 시점: 머신러닝 pkl 모델 로드
    try:
        load_model("app/data/final_hybrid_model.pkl")
    except Exception as e:
        logger.error(f"❌ 힐스트라이크 모델 로드 실패: {e}")
        raise e
        
    yield
    # [2] 서버 종료 시점: 필요한 자원 반납 (있다면 작성)
    logger.info("서버가 종료됩니다.")


# ─────────────────────────────────────────────
# FastAPI 앱 초기화 (lifespan 추가)
# ─────────────────────────────────────────────
app = FastAPI(
    title="AZAS Backend",
    description="IMU 데이터 수집 및 AWS RDS 저장 API + 힐스트라이크 감지",
    version="0.1.0",
    lifespan=lifespan, # 👈 lifespan 연결
)

# DB 테이블 생성 (기존 로직 유지)
Base.metadata.create_all(bind=engine)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포시에는 API Gateway 주소만 넣기
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root():
    return {"message": "Hello Azas!"}