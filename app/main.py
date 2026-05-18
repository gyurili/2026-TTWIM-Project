from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.db.session import engine
from app.db.base import Base
import app.models.imu  # noqa: F401 - Base에 모델 등록

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AZAS Backend",
    description="IMU 데이터 수집 및 AWS RDS 저장 API",
    version="0.1.0",
)

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

