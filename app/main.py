import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text as sa_text

from app.api.router import api_router
from app.api.user import router as user_router
from app.db.session import engine
from app.db.base import Base
from app.core.exceptions import ServiceError
import app.models.imu
import app.models.running
import app.models.dtw
import app.models.chat
import app.models.paper


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with engine.begin() as conn:
            conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as e:
        logger.error("pgvector 확장 활성화 실패 (RAG 챗봇 검색이 동작하지 않을 수 있습니다): %s", e)

    Base.metadata.create_all(bind=engine)

    from app.services.predict import load_models
    try:
        load_models()
    except Exception as e:
        logger.error("모델 로드 실패: %s", e)

    from app.services.paper_rag import load_index
    try:
        load_index()
    except Exception as e:
        logger.error("논문 RAG 인덱스 로드 실패: %s", e)

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
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    message = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
    )
    return JSONResponse(status_code=422, content={"status": "error", "message": message})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("처리되지 않은 예외: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "서버 내부 오류가 발생했습니다."},
    )


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
    return {"status": "success", "data": {"message": "AZAS API"}}
