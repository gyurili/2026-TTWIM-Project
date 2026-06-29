import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.api.user import router as user_router
from app.db.session import engine
from app.db.base import Base
from app.core.exceptions import ServiceError
import app.models.imu


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    from app.services.predict import load_models
    try:
        load_models()
    except Exception as e:
        logger.error("모델 로드 실패: %s", e)

    yield
    logger.info("서버가 종료됩니다.")


app = FastAPI(
    title="AZAS Backend",
    description="IMU 데이터 수집 및 AWS RDS 저장 API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(ServiceError)
async def service_error_handler(request, exc: ServiceError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


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
