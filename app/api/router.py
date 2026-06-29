from fastapi import APIRouter
from app.api import imu
from app.api import predict

api_router = APIRouter()
api_router.include_router(imu.router)
api_router.include_router(predict.router)
