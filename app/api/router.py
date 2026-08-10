from fastapi import APIRouter
from app.api import imu
from app.api import predict
from app.api import running
from app.api import dtw
from app.api import chat

api_router = APIRouter()
api_router.include_router(imu.router)
api_router.include_router(predict.router)
api_router.include_router(running.router)
api_router.include_router(dtw.router)
api_router.include_router(chat.router)
