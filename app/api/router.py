from fastapi import APIRouter
from app.api import imu
from app.api import predict
from app.api import running_pipeline

api_router = APIRouter()
api_router.include_router(imu.router)
api_router.include_router(predict.router)
api_router.include_router(running_pipeline.router)