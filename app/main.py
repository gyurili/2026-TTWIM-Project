import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.user import router as user_router
from app.db.session import engine
from app.db.base import Base
import app.models.imu

from app.services.heelstrike_service import load_model


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model("app/data/final_hybrid_model.pkl")
    except Exception as e:
        logger.error(f"❌ 힐스트라이크 모델 로드 실패: {e}")
        raise e

    yield
    logger.info("서버가 종료됩니다.")


app = FastAPI(
    title="AZAS Backend",
    description="IMU 데이터 수집 및 AWS RDS 저장 API + 힐스트라이크 감지",
    version="0.1.0",
    lifespan=lifespan,
)

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(user_router, prefix="/user")


@app.get("/")
async def root():
    return {"message": "AZAS API"}
