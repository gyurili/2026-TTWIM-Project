import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.db.session import engine
from app.db.base import Base
import app.models.imu


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("서버가 종료됩니다.")

app = FastAPI(
    title="AZAS Backend",
    description="IMU 데이터 수집 및 AWS RDS 저장 API",
    version="0.1.0",
    lifespan=lifespan,
)

# DB 테이블 생성
Base.metadata.create_all(bind=engine)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root():
    return {"message": "Hello Azas!"}