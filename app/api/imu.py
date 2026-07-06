import logging
import uuid
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.core.exceptions import GaitSaveError
from app.db.session import get_db
from app.schemas.common import SuccessResponse
from app.schemas.imu import GaitBatchUpload, GaitBatchResponse
from app.crud import imu as crud
from app.services import imu as service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/imu", tags=["IMU"])


@router.post("/batch", response_model=SuccessResponse[GaitBatchResponse], status_code=201)
def upload_batch(
    payload: GaitBatchUpload,
    db: Session = Depends(get_db),
):
    """React Native에서 JSON 배열로 묶어서 전송"""
    session_id = str(uuid.uuid4())
    try:
        count = crud.bulk_create_gait_records(db, payload.records, session_id)
    except Exception as e:
        logger.error("gait record 저장 실패: %s", e)
        raise GaitSaveError("gait 데이터 저장에 실패했습니다")
    return SuccessResponse(data=GaitBatchResponse(message="저장 완료", count=count))


@router.post("/upload-db", response_model=SuccessResponse[GaitBatchResponse], status_code=201)
async def upload_sqlite_db(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """React Native SQLite .db 파일을 그대로 전송받아 RDS에 저장"""
    session_id = str(uuid.uuid4())
    records = await service.parse_sqlite_upload(file)
    try:
        count = crud.bulk_create_gait_records(db, records, session_id)
    except Exception as e:
        logger.error("gait record 저장 실패: %s", e)
        raise GaitSaveError("gait 데이터 저장에 실패했습니다")
    return SuccessResponse(data=GaitBatchResponse(message="저장 완료", count=count))