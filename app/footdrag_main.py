import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.footdrag import router as footdrag_router
from app.services.footdrag_predict import load_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as exc:
        logger.error("발끌기 모델 로드 실패: %s", exc)
        raise

    yield
    logger.info("발끌기 API 서버가 종료됩니다.")


app = FastAPI(
    title="AZAS Foot Drag API",
    description="정상/발끌기 분류 전용 FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(footdrag_router)


@app.get("/")
def root():
    return {"message": "Azas foot drag API"}
