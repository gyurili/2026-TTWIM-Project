from fastapi import APIRouter
from app.api import imu

api_router = APIRouter()
api_router.include_router(imu.router)